//! Diagnose where time goes in `tip5_permute_sme` test loops.

use acpu::field::tip5::{tip5_permute, tip5_permute_sme};
use std::time::Instant;

fn rs(mut x: u64) -> [u64; 16] {
    let mut s = [0u64; 16];
    for v in s.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    s
}

fn main() {
    if !acpu::probe::scan().has_sme {
        eprintln!("skip: FEAT_SME not present");
        return;
    }

    let n = 1024usize;

    // 1. Cost of just scalar permute, no SME.
    let t0 = Instant::now();
    for i in 0..n {
        let mut s = rs(i as u64 + 1);
        tip5_permute(&mut s);
        std::hint::black_box(&s);
    }
    println!("scalar permute x{n}: {:?}", t0.elapsed());

    // 2. Cost of tip5_permute_sme::<1> in a loop (open Stream every iter).
    let t0 = Instant::now();
    for i in 0..n {
        let mut s = [rs(i as u64 + 1)];
        tip5_permute_sme::<1>(&mut s).unwrap();
        std::hint::black_box(&s);
    }
    println!("tip5_permute_sme::<1> x{n}: {:?}", t0.elapsed());

    // 3. tip5_permute_sme::<N=8> in a loop (amortizes Stream).
    let t0 = Instant::now();
    let iters = n / 8;
    for i in 0..iters {
        let mut s: [[u64; 16]; 8] = core::array::from_fn(|k| rs(i as u64 + k as u64 + 1));
        tip5_permute_sme::<8>(&mut s).unwrap();
        std::hint::black_box(&s);
    }
    println!(
        "tip5_permute_sme::<8> x{iters} ({n} total): {:?}",
        t0.elapsed()
    );

    // 4. Raw Stream::new+drop x N (no permute).
    use acpu::streaming::Stream;
    let t0 = Instant::now();
    for _ in 0..n {
        let s = Stream::new().unwrap();
        std::hint::black_box(&s);
        drop(s);
    }
    println!("Stream::new+drop x{n}: {:?}", t0.elapsed());
}
