fn main() {
    let values = [
        (3.14_f32, f32::from_bits(0x4048_f5c3)),
        (1.266_065_9_f32, 1.266_066_f32),
    ];
    for (old, new) in values {
        assert_eq!(old.to_bits(), new.to_bits());
        println!("{:08x} == {:08x}", old.to_bits(), new.to_bits());
    }
}
