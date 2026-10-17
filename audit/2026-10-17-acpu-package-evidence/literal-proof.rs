fn main() {
    let old_scalar = 3.14159f32;
    let old_s2pi = 0.7978845608f32;
    let new_s2pi = 0.797_884_6_f32;
    assert_eq!(old_s2pi.to_bits(), new_s2pi.to_bits());
    assert_eq!(old_scalar.to_bits(), 0x4049_0fd0);
    assert_eq!(old_scalar, f32::from_bits(0x4049_0fd0));
    println!("scalar={:08x}; s2pi={:08x}; equality=true", old_scalar.to_bits(), old_s2pi.to_bits());
}
