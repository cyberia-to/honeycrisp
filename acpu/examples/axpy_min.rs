fn main() {
    eprintln!("start");
    let caps = acpu::scan();
    if !caps.has_sme {
        return;
    }
    let a = 1.5f32;
    let x = vec![1.0f32; 16];
    let mut y = vec![1.0f32; 16];
    eprintln!("first axpy");
    acpu::streaming::kern::axpy_f32(a, &x, &mut y).unwrap();
    eprintln!("first done: y[0] = {}", y[0]);
}
