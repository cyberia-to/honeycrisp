//! Test AMX pair load alignment requirements.

use acpu::matrix::asm::{amx_clr, amx_op, amx_set, OP_FMA32, OP_LDX, OP_LDY, OP_STZ};

fn main() {
    unsafe {
        amx_set();

        const PAIR: u64 = 1u64 << 62;
        let fma_first: u64 = 1 << 27; // skip_z, X[0], Y[0], tile 0
        let fma_x1: u64 = (1 << 27) | (64 << 10); // X[1], Y[0], tile 0

        #[repr(align(128))]
        struct A128([f32; 64]);

        let y_one = A128({
            let mut a = [0.0f32; 64];
            a[0] = 1.0;
            a
        });
        amx_op::<OP_LDY>(y_one.0.as_ptr() as u64);

        // Test pair LDX at different alignments
        let buf = A128({
            let mut a = [0.0f32; 64];
            for i in 0..64 {
                a[i] = (i + 1) as f32;
            }
            a
        });

        let base = buf.0.as_ptr() as *const u8;

        // Alignment 128 (offset 0)
        let ptr128 = base;
        amx_op::<OP_LDX>((ptr128 as u64) | PAIR);
        amx_op::<OP_FMA32>(fma_x1);
        let mut z = A128([0.0f32; 64]);
        amx_op::<OP_STZ>(z.0.as_mut_ptr() as u64);
        println!("align=128: X[1][0..4] = {:?}", &z.0[0..4]);

        // Alignment 64 (offset 64 bytes = 16 floats)
        let ptr64 = base.add(64);
        amx_op::<OP_LDX>((ptr64 as u64) | PAIR);
        amx_op::<OP_FMA32>(fma_x1);
        amx_op::<OP_STZ>(z.0.as_mut_ptr() as u64);
        println!("align=64:  X[1][0..4] = {:?}", &z.0[0..4]);
        // If works: [33, 34, 35, 36] (offset 32 floats = 128 bytes from base)

        // Alignment 32 (offset 32 bytes = 8 floats) — might fail
        let ptr32 = base.add(32);
        amx_op::<OP_LDX>((ptr32 as u64) | PAIR);
        amx_op::<OP_FMA32>(fma_x1);
        amx_op::<OP_STZ>(z.0.as_mut_ptr() as u64);
        println!("align=32:  X[1][0..4] = {:?}", &z.0[0..4]);

        // Alignment 16 (offset 16 bytes = 4 floats)
        let ptr16 = base.add(16);
        amx_op::<OP_LDX>((ptr16 as u64) | PAIR);
        amx_op::<OP_FMA32>(fma_x1);
        amx_op::<OP_STZ>(z.0.as_mut_ptr() as u64);
        println!("align=16:  X[1][0..4] = {:?}", &z.0[0..4]);

        amx_clr();
    }
}
