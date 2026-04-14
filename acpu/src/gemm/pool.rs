//! Persistent thread pool pinned to P-cores.
//!
//! Workers stay alive across matmul calls, avoiding thread spawn overhead
//! (~20μs per spawn). AMX context is initialized once per worker.

use std::sync::{Arc, Barrier, Mutex};

/// A closure the pool can execute.
type Job = Box<dyn FnOnce() + Send>;

/// Persistent thread pool with pre-spawned, P-core-pinned workers.
pub(super) struct Pool {
    senders: Vec<std::sync::mpsc::Sender<Job>>,
    barrier: Arc<Barrier>,
}

impl Pool {
    /// Create a pool with `n` workers, each pinned to a P-core with AMX active.
    fn new(n: usize) -> Self {
        let barrier = Arc::new(Barrier::new(n + 1));
        let mut senders = Vec::with_capacity(n);

        for _ in 0..n {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            let bar = barrier.clone();
            std::thread::spawn(move || {
                let _ = crate::sync::affinity::pin_p_core();
                super::ensure_amx();
                while let Ok(job) = rx.recv() {
                    job();
                    bar.wait();
                }
            });
            senders.push(tx);
        }

        Self { senders, barrier }
    }

    /// Number of workers.
    pub(super) fn len(&self) -> usize {
        self.senders.len()
    }

    /// Dispatch `jobs.len()` closures to workers and wait for all to complete.
    /// `jobs.len()` must equal `self.len()`.
    pub(super) fn run(&self, jobs: Vec<Job>) {
        assert_eq!(jobs.len(), self.senders.len());
        for (tx, job) in self.senders.iter().zip(jobs) {
            tx.send(job).expect("worker thread panicked");
        }
        self.barrier.wait();
    }
}

// Singleton pool sized to P-core count.
static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);

/// Get or create the global thread pool.
pub(super) fn get_pool(n_threads: usize) -> Arc<Pool> {
    let mut guard = POOL.lock().unwrap();
    if let Some(ref pool) = *guard {
        if pool.len() == n_threads {
            return pool.clone();
        }
    }
    let pool = Arc::new(Pool::new(n_threads));
    *guard = Some(pool.clone());
    pool
}
