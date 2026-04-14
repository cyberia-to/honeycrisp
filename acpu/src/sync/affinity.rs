//! Core affinity via macOS QoS (Quality of Service) classes.
//!
//! Apple Silicon does not expose direct core pinning.  The recommended
//! mechanism is to set the pthread QoS class, which the kernel uses to
//! schedule the thread onto P-cores (high QoS) or E-cores (low QoS).

use crate::CpuError;

// ---------------------------------------------------------------------------
// QoS class constants (from <sys/qos.h>)
// ---------------------------------------------------------------------------

const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
const QOS_CLASS_DEFAULT: u32 = 0x15;
const QOS_CLASS_BACKGROUND: u32 = 0x09;

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Hint the scheduler to run this thread on a P-core.
///
/// Sets QoS to `QOS_CLASS_USER_INTERACTIVE`, the highest priority class,
/// which the kernel maps to performance cores on Apple Silicon.
pub fn pin_p_core() -> crate::Result<()> {
    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    if rc != 0 {
        return Err(CpuError::AffinityFailed(format!(
            "pthread_set_qos_class_self_np(USER_INTERACTIVE) returned {rc}"
        )));
    }
    Ok(())
}

/// Hint the scheduler to run this thread on an E-core.
///
/// Sets QoS to `QOS_CLASS_BACKGROUND`, which the kernel maps to
/// efficiency cores on Apple Silicon.
pub fn pin_e_core() -> crate::Result<()> {
    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0) };
    if rc != 0 {
        return Err(CpuError::AffinityFailed(format!(
            "pthread_set_qos_class_self_np(BACKGROUND) returned {rc}"
        )));
    }
    Ok(())
}

/// Reset the scheduler hint to the default (no core preference).
///
/// Sets QoS to `QOS_CLASS_DEFAULT`, letting the kernel decide freely.
pub fn pin_any() -> crate::Result<()> {
    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0) };
    if rc != 0 {
        return Err(CpuError::AffinityFailed(format!(
            "pthread_set_qos_class_self_np(DEFAULT) returned {rc}"
        )));
    }
    Ok(())
}
