//! Bit-identity gate: `tip5_permute_sme<N>` must produce the exact same
//! state as N independent scalar `tip5_permute` calls. Non-negotiable per
//! the M4 SME upgrade plan.

use acpu::field::tip5::{tip5_permute, tip5_permute_sme};
use rand::{RngCore, SeedableRng};

const N_SAMPLES: usize = 1024;

fn random_state(rng: &mut impl RngCore) -> [u64; 16] {
    let mut s = [0u64; 16];
    for v in s.iter_mut() {
        *v = rng.next_u64();
    }
    s
}

fn check_skip() -> bool {
    if !acpu::probe::scan().has_sme {
        eprintln!("skip: FEAT_SME not present");
        return true;
    }
    false
}

#[test]
fn sme_permute_n1_property() {
    if check_skip() {
        return;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA001);
    for i in 0..N_SAMPLES {
        let s0 = random_state(&mut rng);
        let mut a = [s0];
        let mut b = s0;
        tip5_permute_sme::<1>(&mut a).expect("Stream::new");
        tip5_permute(&mut b);
        assert_eq!(a[0], b, "iter {i} (N=1)");
    }
}

#[test]
fn sme_permute_n4_property() {
    if check_skip() {
        return;
    }
    const N: usize = 4;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA004);
    let iters = N_SAMPLES / N;
    for i in 0..iters {
        let mut a: [[u64; 16]; N] = core::array::from_fn(|_| random_state(&mut rng));
        let mut b = a;
        tip5_permute_sme::<N>(&mut a).expect("Stream::new");
        for slot in b.iter_mut() {
            tip5_permute(slot);
        }
        for (k, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(sa, sb, "iter {i}, slot {k}");
        }
    }
}

#[test]
fn sme_permute_n8_property() {
    if check_skip() {
        return;
    }
    const N: usize = 8;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA008);
    let iters = N_SAMPLES / N;
    for i in 0..iters {
        let mut a: [[u64; 16]; N] = core::array::from_fn(|_| random_state(&mut rng));
        let mut b = a;
        tip5_permute_sme::<N>(&mut a).expect("Stream::new");
        for slot in b.iter_mut() {
            tip5_permute(slot);
        }
        for (k, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(sa, sb, "iter {i}, slot {k}");
        }
    }
}

#[test]
fn sme_permute_n16_property() {
    if check_skip() {
        return;
    }
    const N: usize = 16;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xA016);
    let iters = N_SAMPLES / N;
    for i in 0..iters {
        let mut a: [[u64; 16]; N] = core::array::from_fn(|_| random_state(&mut rng));
        let mut b = a;
        tip5_permute_sme::<N>(&mut a).expect("Stream::new");
        for slot in b.iter_mut() {
            tip5_permute(slot);
        }
        for (k, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(sa, sb, "iter {i}, slot {k}");
        }
    }
}
