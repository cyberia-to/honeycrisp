#[path = "../../../acpu/bench/common.rs"]
mod common;

fn main() {
    assert_eq!(std::mem::size_of::<i64>(), 8);
    assert_eq!(std::mem::size_of::<u64>(), 8);
    let a = [1.0_f32, 2.0, 3.0];
    let b = [10.0_f32, 20.0, 30.0];
    let mut out = [0.0_f32; 3];
    let mut sum = 0.0_f32;
    // Both input pointers begin at valid elements. A traverses positions
    // 2,1,0 with stride -1; B/output traverse positions 0,1,2 with stride 1.
    unsafe {
        common::vDSP_vadd(a.as_ptr(), 1, b.as_ptr(), 1, out.as_mut_ptr(), 1, 3);
        assert_eq!(out, [11.0, 22.0, 33.0]);
        common::vDSP_vadd(a.as_ptr().add(2), -1, b.as_ptr(), 1, out.as_mut_ptr(), 1, 3);
        assert_eq!(out, [13.0, 22.0, 31.0]);
        common::vDSP_vmul(a.as_ptr().add(2), -1, b.as_ptr(), 1, out.as_mut_ptr(), 1, 3);
        assert_eq!(out, [30.0, 40.0, 30.0]);
        common::vDSP_sve(a.as_ptr().add(2), -1, &mut sum, 3);
        assert_eq!(sum, 6.0);
    }
    println!("shared Rust declarations: forward-add={{11,22,33}}, reverse-add={{13,22,31}}, reverse-mul={{30,40,30}}, reverse-sum=6");
}
