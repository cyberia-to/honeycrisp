//! ShaderLib + Shader: shader compilation from Metal Shading Language

use crate::ffi::*;
use crate::GpuError;

/// A compiled shader library. Wraps `id<MTLLibrary>`.
pub struct ShaderLib {
    raw: ObjcId,
}

impl ShaderLib {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        ShaderLib { raw }
    }

    /// Get a function by name from the library.
    pub fn function(&self, name: &str) -> Result<Shader, GpuError> {
        let ns_name = nsstring(name);
        let raw = unsafe {
            let r = msg1::<ObjcId>(self.raw, SEL_newFunctionWithName(), ns_name);
            CFRelease(ns_name as CFTypeRef);
            r
        };
        if raw.is_null() {
            return Err(GpuError::FunctionNotFound(name.into()));
        }
        Ok(Shader { raw })
    }

    /// List all function names in the library.
    pub fn function_names(&self) -> Vec<String> {
        unsafe {
            let arr = msg0(self.raw, SEL_functionNames());
            if arr.is_null() {
                return vec![];
            }
            let count = msg0_usize(arr, SEL_count());
            let mut names = Vec::with_capacity(count);
            for i in 0..count {
                let ns = msg1::<NSUInteger>(arr, SEL_objectAtIndex(), i);
                if let Some(s) = nsstring_to_rust(ns) {
                    names.push(s);
                }
            }
            names
        }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for ShaderLib {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A single shader function. Wraps `id<MTLFunction>`.
pub struct Shader {
    raw: ObjcId,
}

impl Shader {
    pub fn name(&self) -> String {
        unsafe {
            let ns = msg0(self.raw, SEL_name());
            nsstring_to_rust(ns).unwrap_or_else(|| "unknown".into())
        }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for Shader {
    #[mutants::skip] // RAII release — tested by drop_stress_leak
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
