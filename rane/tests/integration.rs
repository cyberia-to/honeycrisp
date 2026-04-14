//! Integration tests — no ANE hardware required
//! These tests verify fp16 conversion, MIL generation, and IOSurface creation.
//! Run on any macOS (including Intel CI runners).

use rane::{f32_to_fp16, fp16_to_f32, Buffer};

// ── Surface tests ──

#[test]
fn surface_create_and_size() {
    let s = Buffer::new(4096).unwrap();
    assert_eq!(s.size(), 4096, "exact size for 4096 bytes");
    assert!(s.id() > 0);
    // two surfaces must have different ids
    let s2 = Buffer::new(4096).unwrap();
    assert_ne!(s.id(), s2.id(), "surface ids must be unique");
}

#[test]
fn surface_with_shape() {
    // with_shape(ch, sp) must allocate exactly ch * sp * 2 bytes (fp16)
    let s = Buffer::with_shape(64, 128).unwrap();
    assert_eq!(s.size(), 64 * 128 * 2, "with_shape size must be ch*sp*2");
    // different shapes produce different sizes
    let s2 = Buffer::with_shape(32, 64).unwrap();
    assert_eq!(s2.size(), 32 * 64 * 2);
    assert_ne!(s.size(), s2.size());
}

#[test]
fn surface_write_read_roundtrip() {
    let n_elements = 256usize;
    let s = Buffer::new(n_elements * 2).unwrap();
    // verify slice length matches expected element count
    s.write(|d| {
        assert_eq!(d.len(), n_elements, "write slice length must be size/2");
        for i in 0..n_elements {
            d[i] = f32_to_fp16(i as f32);
        }
    });
    s.read(|d| {
        assert_eq!(d.len(), n_elements, "read slice length must be size/2");
        for i in 0..n_elements {
            let val = fp16_to_f32(d[i]);
            assert!((val - i as f32).abs() < 0.5, "mismatch at {}: {}", i, val);
        }
    });
}

#[test]
fn surface_bounds_check() {
    assert!(Buffer::new(0).is_err());
    assert!(Buffer::new(512 * 1024 * 1024).is_err()); // > 256 MB limit
}

// ── fp16 tests ──

#[test]
fn fp16_roundtrip_exact() {
    for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0] {
        let h = f32_to_fp16(v);
        let back = fp16_to_f32(h);
        assert_eq!(back, v, "roundtrip failed for {}", v);
    }
}

#[test]
fn fp16_bulk_conversion() {
    let n = 1024;
    let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut fp16 = vec![0u16; n];
    let mut dst = vec![0.0f32; n];

    rane::cast_f32_f16(&mut fp16, &src);
    rane::cast_f16_f32(&mut dst, &fp16);

    for i in 0..n {
        let err = (dst[i] - src[i]).abs();
        assert!(
            err < 0.1,
            "bulk conversion error at {}: {} vs {}",
            i,
            dst[i],
            src[i]
        );
    }
}

// ── MIL program tests ──

#[test]
fn mil_matmul_shape() {
    let p = rane::mil::matmul(64, 64, 64);
    assert_eq!(p.input_shape(), (64, 128));
    assert_eq!(p.output_shape(), (64, 64));
    assert_eq!(p.input_size(), 64 * 128 * 2);
    assert_eq!(p.output_size(), 64 * 64 * 2);
}

#[test]
fn mil_matmul_text_is_valid() {
    let p = rane::mil::matmul(32, 16, 8);
    let text = p.as_str();
    assert!(text.contains("program(1.3)"));
    assert!(text.contains("func main"));
    assert!(text.contains("matmul"));
}

// ── pack_weights tests ──

#[test]
fn pack_weights_length_and_header() {
    let data: Vec<u16> = (0..128).collect();
    let blob = rane::pack_weights(&data);

    // output length = 128-byte header + input_bytes
    let input_bytes = data.len() * 2;
    assert_eq!(blob.len(), 128 + input_bytes);

    // magic at 0x40..0x44 = 0xDEADBEEF (little-endian)
    assert_eq!(blob[0x40], 0xEF);
    assert_eq!(blob[0x41], 0xBE);
    assert_eq!(blob[0x42], 0xAD);
    assert_eq!(blob[0x43], 0xDE);

    // dtype at 0x44 = 1 (fp16)
    assert_eq!(blob[0x44], 1);

    // data after header matches input
    for (i, &val) in data.iter().enumerate() {
        let off = 128 + i * 2;
        let stored = u16::from_le_bytes([blob[off], blob[off + 1]]);
        assert_eq!(stored, val, "mismatch at index {}", i);
    }
}

#[test]
fn pack_weights_empty() {
    let blob = rane::pack_weights(&[]);
    assert_eq!(blob.len(), 128);
}

// ── mil_header / mil_footer / gen_dyn_matmul tests ──

#[test]
fn mil_header_contains_keywords() {
    let h = rane::mil_header(64, 128);
    assert!(!h.is_empty());
    assert!(h.contains("program(1.3)"));
    assert!(h.contains("func main"));
    assert!(h.contains("fp16"));
    assert!(h.contains("[1, 64, 1, 128]"));
}

#[test]
fn mil_footer_contains_closing() {
    let f = rane::mil_footer("out_var");
    assert!(!f.is_empty());
    assert!(f.contains("out_var"));
    assert!(f.contains("}"));
}

#[test]
fn gen_dyn_matmul_produces_mil_ops() {
    let mut m = String::new();
    rane::gen_dyn_matmul(&mut m, "test", 32, 16, 8, 0, 8, "x");
    assert!(!m.is_empty());
    assert!(m.contains("matmul"));
    assert!(m.contains("slice_by_size"));
    assert!(m.contains("reshape"));
    assert!(m.contains("transpose"));
    assert!(m.contains("test_y"));
}

// ── AneError Display tests ──

#[test]
fn ane_error_display_variants() {
    use rane::AneError;

    let variants: Vec<AneError> = vec![
        AneError::SurfaceCreationFailed("test".into()),
        AneError::ClassNotFound("TestClass"),
        AneError::DescriptorCreationFailed,
        AneError::ModelCreationFailed,
        AneError::CompilationFailed("bad mil".into()),
        AneError::LoadFailed("no ane".into()),
        AneError::EvalFailed("shape mismatch".into()),
        AneError::UnloadFailed("busy".into()),
        AneError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "missing")),
    ];

    for v in &variants {
        let s = format!("{}", v);
        assert!(!s.is_empty(), "Display for {:?} was empty", v);
    }
}

// ── Error path tests ──

#[test]
fn buffer_zero_size_is_err() {
    assert!(Buffer::new(0).is_err());
}

#[test]
fn buffer_oversized_is_err() {
    // 512 MB > 256 MB limit
    assert!(Buffer::new(512 * 1024 * 1024).is_err());
}

// ── Source has no from_text constructor ──
// Source only has matmul() as a public constructor; no from_text exists.
