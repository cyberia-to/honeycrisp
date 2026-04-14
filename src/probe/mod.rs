//! Chip detection and capability probing for Apple Silicon M1–M4.
//!
//! Uses raw `sysctlbyname` calls — no external dependencies.

use crate::CpuError;
use std::ffi::CStr;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// FFI: sysctlbyname from libSystem
// ---------------------------------------------------------------------------

extern "C" {
    fn sysctlbyname(
        name: *const u8,
        oldp: *mut u8,
        oldlenp: *mut usize,
        newp: *const u8,
        newlen: usize,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// Chip enum
// ---------------------------------------------------------------------------

/// Apple Silicon chip variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Chip {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    Unknown,
}

impl Chip {
    /// Identify the chip from sysctl strings.
    ///
    /// Tries `machdep.cpu.brand_string` first, then falls back to
    /// the product identifier from `hw.product`.
    pub fn detect() -> Chip {
        // Try brand string first (available on macOS 13+)
        if let Ok(brand) = sysctl_string(b"machdep.cpu.brand_string\0") {
            if let Some(c) = Self::parse_brand(&brand) {
                return c;
            }
        }
        // Fallback: hw.product (e.g. "MacBookPro18,1")
        if let Ok(product) = sysctl_string(b"hw.product\0") {
            if let Some(c) = Self::parse_product(&product) {
                return c;
            }
        }
        // Last resort: check hw.optional.arm.FEAT_SME — only M4 family has it
        if sysctl_u32(b"hw.optional.arm.FEAT_SME\0").unwrap_or(0) != 0 {
            return Chip::M4;
        }
        Chip::Unknown
    }

    fn parse_brand(brand: &str) -> Option<Chip> {
        let b = brand.to_ascii_lowercase();
        // Match most specific first (Ultra before plain).
        if b.contains("m1 ultra") {
            Some(Chip::M1Ultra)
        } else if b.contains("m1 max") {
            Some(Chip::M1Max)
        } else if b.contains("m1 pro") {
            Some(Chip::M1Pro)
        } else if b.contains("m1") {
            Some(Chip::M1)
        } else if b.contains("m2 ultra") {
            Some(Chip::M2Ultra)
        } else if b.contains("m2 max") {
            Some(Chip::M2Max)
        } else if b.contains("m2 pro") {
            Some(Chip::M2Pro)
        } else if b.contains("m2") {
            Some(Chip::M2)
        } else if b.contains("m3 ultra") {
            Some(Chip::M3Ultra)
        } else if b.contains("m3 max") {
            Some(Chip::M3Max)
        } else if b.contains("m3 pro") {
            Some(Chip::M3Pro)
        } else if b.contains("m3") {
            Some(Chip::M3)
        } else if b.contains("m4 max") {
            Some(Chip::M4Max)
        } else if b.contains("m4 pro") {
            Some(Chip::M4Pro)
        } else if b.contains("m4") {
            Some(Chip::M4)
        } else {
            None
        }
    }

    /// Coarse chip-generation detection from product board IDs.
    fn parse_product(product: &str) -> Option<Chip> {
        // Board IDs like "MacBookPro18,3" (M1 Pro), "Mac14,6" (M2 Max) etc.
        // We only map the generation; the exact SKU variant needs brand_string.
        let p = product.to_ascii_lowercase();
        if p.starts_with("virtualm") || p.starts_with("apple m") {
            // VM environments sometimes expose the chip name directly.
            return Self::parse_brand(product);
        }
        // Without a full product-id table, return Unknown and let other
        // heuristics (SME feature flag etc.) refine later.
        None
    }

    /// AMX version supported by this chip generation.
    pub fn amx_version(self) -> u8 {
        match self {
            Chip::M1 | Chip::M1Pro | Chip::M1Max | Chip::M1Ultra => 1,
            Chip::M2 | Chip::M2Pro | Chip::M2Max | Chip::M2Ultra => 1,
            Chip::M3 | Chip::M3Pro | Chip::M3Max | Chip::M3Ultra => 2,
            Chip::M4 | Chip::M4Pro | Chip::M4Max => 2,
            Chip::Unknown => 0,
        }
    }
}

impl std::fmt::Display for Chip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Chip::M1 => "Apple M1",
            Chip::M1Pro => "Apple M1 Pro",
            Chip::M1Max => "Apple M1 Max",
            Chip::M1Ultra => "Apple M1 Ultra",
            Chip::M2 => "Apple M2",
            Chip::M2Pro => "Apple M2 Pro",
            Chip::M2Max => "Apple M2 Max",
            Chip::M2Ultra => "Apple M2 Ultra",
            Chip::M3 => "Apple M3",
            Chip::M3Pro => "Apple M3 Pro",
            Chip::M3Max => "Apple M3 Max",
            Chip::M3Ultra => "Apple M3 Ultra",
            Chip::M4 => "Apple M4",
            Chip::M4Pro => "Apple M4 Pro",
            Chip::M4Max => "Apple M4 Max",
            Chip::Unknown => "Unknown",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Feature enum
// ---------------------------------------------------------------------------

/// ARM CPU feature extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Half-precision floating point (FP16 / FEAT_FP16)
    Fp16,
    /// BFloat16 (FEAT_BF16)
    Bf16,
    /// Advanced SIMD dot-product (FEAT_DotProd)
    DotProd,
    /// Int8 matrix multiply (FEAT_I8MM)
    I8mm,
    /// Floating-point complex multiply-add (FEAT_FCMA)
    Fcma,
    /// Rounding double multiply-add (FEAT_RDM)
    Rdm,
    /// Large System Extensions — atomics (FEAT_LSE)
    Lse,
    /// Load-acquire RCpc (FEAT_LRCPC)
    Lrcpc,
}

// ---------------------------------------------------------------------------
// Features struct
// ---------------------------------------------------------------------------

/// Detected hardware capabilities (cached, immutable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    pub chip: Chip,
    /// AMX coprocessor version (0 = not detected, 1 = M1/M2, 2 = M3/M4)
    pub amx_ver: u8,
    pub has_fp16: bool,
    pub has_bf16: bool,
    pub has_dotprod: bool,
    pub has_i8mm: bool,
    pub has_fcma: bool,
    pub has_rdm: bool,
    pub has_lse: bool,
    pub has_lrcpc: bool,
    /// Performance (P) cores
    pub p_cores: u8,
    /// Efficiency (E) cores
    pub e_cores: u8,
    /// L1 cache line size in bytes
    pub l1_line: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
}

impl Features {
    /// Check whether a given CPU feature is available.
    pub fn has(&self, feat: Feature) -> bool {
        match feat {
            Feature::Fp16 => self.has_fp16,
            Feature::Bf16 => self.has_bf16,
            Feature::DotProd => self.has_dotprod,
            Feature::I8mm => self.has_i8mm,
            Feature::Fcma => self.has_fcma,
            Feature::Rdm => self.has_rdm,
            Feature::Lse => self.has_lse,
            Feature::Lrcpc => self.has_lrcpc,
        }
    }
}

// ---------------------------------------------------------------------------
// sysctl helpers
// ---------------------------------------------------------------------------

/// Query a sysctl key that returns a `u32`.
fn sysctl_u32(name: &[u8]) -> Result<u32, CpuError> {
    let mut val: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u32 as *mut u8,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 {
        let key = CStr::from_bytes_with_nul(name)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("<invalid>"));
        return Err(CpuError::SysctlFailed(key));
    }
    Ok(val)
}

/// Query a sysctl key that returns a `u64`.
fn sysctl_u64(name: &[u8]) -> Result<u64, CpuError> {
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u64 as *mut u8,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 {
        let key = CStr::from_bytes_with_nul(name)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("<invalid>"));
        return Err(CpuError::SysctlFailed(key));
    }
    Ok(val)
}

/// Query a sysctl key that returns a C string.
fn sysctl_string(name: &[u8]) -> Result<String, CpuError> {
    // First call: get the required buffer length.
    let mut len: usize = 0;
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        let key = CStr::from_bytes_with_nul(name)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("<invalid>"));
        return Err(CpuError::SysctlFailed(key));
    }

    let mut buf = vec![0u8; len];
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr(),
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc != 0 {
        let key = CStr::from_bytes_with_nul(name)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from("<invalid>"));
        return Err(CpuError::SysctlFailed(key));
    }

    // Strip trailing NUL.
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
    String::from_utf8(buf).map_err(|e| CpuError::SysctlFailed(e.to_string()))
}

/// Check a `hw.optional.arm.FEAT_*` flag (returns `false` on any error).
fn feat_flag(name: &[u8]) -> bool {
    sysctl_u32(name).unwrap_or(0) != 0
}

// ---------------------------------------------------------------------------
// Core-count helpers
// ---------------------------------------------------------------------------

fn detect_p_cores() -> u8 {
    sysctl_u32(b"hw.perflevel0.physicalcpu\0").unwrap_or(0) as u8
}

fn detect_e_cores() -> u8 {
    sysctl_u32(b"hw.perflevel1.physicalcpu\0").unwrap_or(0) as u8
}

fn detect_l1_line() -> usize {
    sysctl_u64(b"hw.cachelinesize\0").unwrap_or(128) as usize
}

fn detect_l2_size() -> usize {
    sysctl_u64(b"hw.l2cachesize\0").unwrap_or(0) as usize
}

// ---------------------------------------------------------------------------
// Feature detection
// ---------------------------------------------------------------------------

fn detect_features() -> (bool, bool, bool, bool, bool, bool, bool, bool) {
    let fp16 = feat_flag(b"hw.optional.arm.FEAT_FP16\0");
    let bf16 = feat_flag(b"hw.optional.arm.FEAT_BF16\0");
    let dotprod = feat_flag(b"hw.optional.arm.FEAT_DotProd\0");
    let i8mm = feat_flag(b"hw.optional.arm.FEAT_I8MM\0");
    let fcma = feat_flag(b"hw.optional.arm.FEAT_FCMA\0");
    let rdm = feat_flag(b"hw.optional.arm.FEAT_RDM\0");
    let lse = feat_flag(b"hw.optional.arm.FEAT_LSE\0");
    let lrcpc = feat_flag(b"hw.optional.arm.FEAT_LRCPC\0");
    (fp16, bf16, dotprod, i8mm, fcma, rdm, lse, lrcpc)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

static CAPS: OnceLock<Features> = OnceLock::new();

/// Detect hardware capabilities. The result is computed once and cached for
/// the lifetime of the process.
pub fn scan() -> &'static Features {
    CAPS.get_or_init(|| {
        let chip = Chip::detect();
        let amx_ver = chip.amx_version();
        let (fp16, bf16, dotprod, i8mm, fcma, rdm, lse, lrcpc) = detect_features();

        Features {
            chip,
            amx_ver,
            has_fp16: fp16,
            has_bf16: bf16,
            has_dotprod: dotprod,
            has_i8mm: i8mm,
            has_fcma: fcma,
            has_rdm: rdm,
            has_lse: lse,
            has_lrcpc: lrcpc,
            p_cores: detect_p_cores(),
            e_cores: detect_e_cores(),
            l1_line: detect_l1_line(),
            l2_size: detect_l2_size(),
        }
    })
}

/// Convenience: return the detected chip variant.
pub fn chip() -> Chip {
    scan().chip
}

/// Convenience: check whether a CPU feature is available.
pub fn has(feat: Feature) -> bool {
    scan().has(feat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_brand_m1() {
        assert_eq!(Chip::parse_brand("Apple M1"), Some(Chip::M1));
        assert_eq!(Chip::parse_brand("Apple M1 Pro"), Some(Chip::M1Pro));
        assert_eq!(Chip::parse_brand("Apple M1 Max"), Some(Chip::M1Max));
        assert_eq!(Chip::parse_brand("Apple M1 Ultra"), Some(Chip::M1Ultra));
    }

    #[test]
    fn parse_brand_m4() {
        assert_eq!(Chip::parse_brand("Apple M4"), Some(Chip::M4));
        assert_eq!(Chip::parse_brand("Apple M4 Pro"), Some(Chip::M4Pro));
        assert_eq!(Chip::parse_brand("Apple M4 Max"), Some(Chip::M4Max));
    }

    #[test]
    fn parse_brand_unknown() {
        assert_eq!(Chip::parse_brand("Intel Core i9"), None);
    }

    #[test]
    fn amx_versions() {
        assert_eq!(Chip::M1.amx_version(), 1);
        assert_eq!(Chip::M2Ultra.amx_version(), 1);
        assert_eq!(Chip::M3.amx_version(), 2);
        assert_eq!(Chip::M4Max.amx_version(), 2);
        assert_eq!(Chip::Unknown.amx_version(), 0);
    }

    #[test]
    fn caps_has_roundtrip() {
        let caps = Features {
            chip: Chip::M1,
            amx_ver: 1,
            has_fp16: true,
            has_bf16: false,
            has_dotprod: true,
            has_i8mm: false,
            has_fcma: true,
            has_rdm: true,
            has_lse: true,
            has_lrcpc: true,
            p_cores: 4,
            e_cores: 4,
            l1_line: 128,
            l2_size: 12 * 1024 * 1024,
        };
        assert!(caps.has(Feature::Fp16));
        assert!(!caps.has(Feature::Bf16));
        assert!(caps.has(Feature::DotProd));
        assert!(!caps.has(Feature::I8mm));
    }

    #[test]
    fn scan_returns_consistent() {
        // Calling scan twice must yield the same pointer (OnceLock).
        let a = scan() as *const Features;
        let b = scan() as *const Features;
        assert_eq!(a, b);
    }

    #[test]
    fn chip_display() {
        assert_eq!(format!("{}", Chip::M3Pro), "Apple M3 Pro");
        assert_eq!(format!("{}", Chip::Unknown), "Unknown");
    }
}
