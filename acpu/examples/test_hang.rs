fn main() {
    let sizes: &[usize] = &[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

    for &sz in sizes {
        let a = vec![1.0f32; sz * sz];
        let b = vec![1.0f32; sz * sz];
        let mut c = vec![0.0f32; sz * sz];
        eprint!("{:>5}...", sz);
        acpu::matmul_f32_set(&a, &b, &mut c, sz, sz, sz);
        eprintln!(" ok (c[0]={})", c[0]);
    }
}
