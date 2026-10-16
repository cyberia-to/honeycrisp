fn main() {
    let decimal = 3.14_f32;
    let frozen = f32::from_bits(0x4048_f5c3);
    assert_eq!(decimal.to_bits(), frozen.to_bits());
    println!("3.14 f32 = from_bits(0x{:08x})", decimal.to_bits());
    let previous = 1.266_065_9_f32;
    let concise = 1.266_066_f32;
    assert_eq!(previous.to_bits(), concise.to_bits());
    println!("1.266_065_9 f32 = 1.266_066 f32 = 0x{:08x}", previous.to_bits());
}
