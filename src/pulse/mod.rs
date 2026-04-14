//! PMU (Performance Monitoring Unit) counters via Apple's kpc interface.
//!
//! Requires running as root **or** with the `dtrace_proc` entitlement
//! (the Instruments profiler uses the same mechanism).  On stock macOS
//! without SIP changes this will fail with [`crate::CpuError::PmuNotAvailable`].

pub mod ffi;

use crate::CpuError;
use ffi::{KPC_CLASS_CONFIGURABLE, KPC_CLASS_FIXED};

// ---------------------------------------------------------------------------
// Counter enum
// ---------------------------------------------------------------------------

/// Which hardware event to count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Counter {
    Cycles = 0,
    Instructions = 1,
    Branches = 2,
    BranchMisses = 3,
    L1dMisses = 4,
    L1iMisses = 5,
    L2Misses = 6,
}

impl Counter {
    /// Number of defined counter types.
    pub const COUNT: usize = 7;
}

// ---------------------------------------------------------------------------
// Snapshot / Counts
// ---------------------------------------------------------------------------

/// Raw counter values captured at a single point in time.
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub counters: [u64; 16],
}

/// Human-readable delta between two [`Snapshot`]s.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub cycles: u64,
    pub instructions: u64,
    pub branches: u64,
    pub branch_misses: u64,
    pub l1d_misses: u64,
    pub l1i_misses: u64,
    pub l2_misses: u64,
}

// ---------------------------------------------------------------------------
// PMU event numbers (Apple Silicon, may vary by generation)
// ---------------------------------------------------------------------------

/// Map our Counter enum to the Apple PMU event selector.
///
/// These selectors are for M1–M4; they live in configurable counter
/// config words as `(event << 0)`.  Fixed counters (0 = cycles,
/// 1 = instructions) are always present.
fn event_selector(c: Counter) -> u64 {
    match c {
        Counter::Cycles => 0x02,       // FIXED_CYCLES (also config 0x02)
        Counter::Instructions => 0x8c, // FIXED_INSTRUCTIONS (also 0x8c)
        Counter::Branches => 0x90,     // INST_BRANCH
        Counter::BranchMisses => 0x91, // BRANCH_MISPRED_NONSPEC
        Counter::L1dMisses => 0xa3,    // L1D_CACHE_MISS_LD_NONSPEC
        Counter::L1iMisses => 0xa1,    // L1I_CACHE_MISS_DEMAND
        Counter::L2Misses => 0xa8,     // L2_CACHE_MISS_DEMAND
    }
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

/// Handle to an active PMU counting session.
///
/// # Example
///
/// ```no_run
/// use acpu::pulse::{Counter, Counters};
///
/// let mut ctx = Counters::new(&[Counter::Cycles, Counter::Instructions]).unwrap();
/// ctx.start();
/// // ... workload ...
/// let a = ctx.read();
/// // ... more workload ...
/// let b = ctx.read();
/// ctx.stop();
/// let c = ctx.elapsed(&a, &b);
/// println!("IPC = {:.2}", c.instructions as f64 / c.cycles as f64);
/// ```
pub struct Counters {
    classes: u32,
    counter_count: u32,
    /// Which of our Counter variants are active, in order.
    active: Vec<Counter>,
    /// Mapping: active[i] -> hardware counter index.
    hw_indices: Vec<usize>,
}

impl Counters {
    /// Configure the PMU with the requested counters.
    ///
    /// The first two fixed counters (cycles, instructions) are always
    /// enabled.  Additional counters use the configurable PMU slots.
    pub fn new(counters: &[Counter]) -> crate::Result<Self> {
        let vt = ffi::vtable().map_err(|_| CpuError::PmuNotAvailable)?;

        let classes = KPC_CLASS_FIXED | KPC_CLASS_CONFIGURABLE;
        let n_counters = unsafe { (vt.get_counter_count)(classes) };
        let n_config = unsafe { (vt.get_config_count)(classes) };

        // Build config array: fixed counters need no config, configurable
        // counters get event selectors.
        let mut config = vec![0u64; n_config as usize];

        // Fixed counters occupy the first slots; configurable follow.
        let n_fixed = unsafe { (vt.get_counter_count)(KPC_CLASS_FIXED) } as usize;
        let mut cfg_slot = 0usize;

        let mut active = Vec::new();
        let mut hw_indices = Vec::new();

        for &c in counters {
            match c {
                // Fixed counter 0 is cycles, 1 is instructions on Apple Silicon.
                Counter::Cycles => {
                    active.push(c);
                    hw_indices.push(0);
                }
                Counter::Instructions => {
                    active.push(c);
                    hw_indices.push(1);
                }
                _ => {
                    if n_fixed + cfg_slot >= n_counters as usize {
                        return Err(CpuError::PmuConfigFailed(
                            "not enough configurable PMU slots".into(),
                        ));
                    }
                    if cfg_slot < config.len() {
                        config[cfg_slot] = event_selector(c);
                    }
                    active.push(c);
                    hw_indices.push(n_fixed + cfg_slot);
                    cfg_slot += 1;
                }
            }
        }

        let rc = unsafe { (vt.set_config)(classes, config.as_ptr()) };
        if rc != 0 {
            return Err(CpuError::PmuConfigFailed(format!(
                "kpc_set_config returned {rc}"
            )));
        }

        Ok(Self {
            classes,
            counter_count: n_counters,
            active,
            hw_indices,
        })
    }

    /// Enable counting on the current thread.
    pub fn start(&mut self) {
        let vt = ffi::vtable().expect("vtable already resolved");
        unsafe {
            (vt.set_counting)(self.classes);
            (vt.set_thread_counting)(self.classes);
        }
    }

    /// Read the current counter values into a [`Snapshot`].
    pub fn read(&self) -> Snapshot {
        let vt = ffi::vtable().expect("vtable already resolved");
        let mut snap = Snapshot::default();
        unsafe {
            (vt.get_thread_counters)(
                0, // current thread
                self.counter_count.min(16),
                snap.counters.as_mut_ptr(),
            );
        }
        snap
    }

    /// Disable counting.
    pub fn stop(&mut self) {
        let vt = ffi::vtable().expect("vtable already resolved");
        unsafe {
            (vt.set_thread_counting)(0);
            (vt.set_counting)(0);
        }
    }

    /// Compute the delta between two snapshots for each active counter.
    pub fn elapsed(&self, a: &Snapshot, b: &Snapshot) -> Counts {
        let mut counts = Counts::default();
        for (i, &c) in self.active.iter().enumerate() {
            let idx = self.hw_indices[i];
            let delta = b.counters[idx].wrapping_sub(a.counters[idx]);
            match c {
                Counter::Cycles => counts.cycles = delta,
                Counter::Instructions => counts.instructions = delta,
                Counter::Branches => counts.branches = delta,
                Counter::BranchMisses => counts.branch_misses = delta,
                Counter::L1dMisses => counts.l1d_misses = delta,
                Counter::L1iMisses => counts.l1i_misses = delta,
                Counter::L2Misses => counts.l2_misses = delta,
            }
        }
        counts
    }
}
