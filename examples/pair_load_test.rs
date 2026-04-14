//! Minimal test: what does bit 62 in LDX/LDY actually do?
//! Load known data, store back, compare.

fn main() {
    unsafe {
        acpu::matrix::asm::amx_set();

        // 256-byte aligned buffer with known pattern
        #[repr(align(256))]
        struct A256([f32; 64]);

        let src = A256([0.0; 64]);
        let src_ptr = src.0.as_ptr() as *mut f32;
        // Fill: src[0..15] = 1.0, src[16..31] = 2.0, src[32..47] = 3.0, src[48..63] = 4.0
        for i in 0..16 {
            *src_ptr.add(i) = 1.0;
        }
        for i in 16..32 {
            *src_ptr.add(i) = 2.0;
        }
        for i in 32..48 {
            *src_ptr.add(i) = 3.0;
        }
        for i in 48..64 {
            *src_ptr.add(i) = 4.0;
        }

        let mut dst0 = A256([0.0; 64]);
        let mut dst1 = A256([0.0; 64]);

        // Test 1: Normal LDX into X[0], then STX X[0]
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_LDX }>(src_ptr as u64);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STX }>(dst0.0.as_mut_ptr() as u64);
        eprintln!(
            "Normal LDX X[0]: first 4 = [{}, {}, {}, {}]",
            dst0.0[0], dst0.0[1], dst0.0[2], dst0.0[3]
        );

        // Test 2: LDX with bit 62 set, then check X[0] and X[1]
        dst0.0.fill(0.0);
        dst1.0.fill(0.0);
        let pair_addr = (src_ptr as u64) | (1u64 << 62);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_LDX }>(pair_addr);
        // Store X[0]
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STX }>(dst0.0.as_mut_ptr() as u64);
        // Store X[1]
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STX }>(
            (dst1.0.as_mut_ptr() as u64) | (1u64 << 56),
        );

        eprintln!("Pair LDX (bit62):");
        eprintln!(
            "  X[0] first 4: [{}, {}, {}, {}]",
            dst0.0[0], dst0.0[1], dst0.0[2], dst0.0[3]
        );
        eprintln!(
            "  X[1] first 4: [{}, {}, {}, {}]",
            dst1.0[0], dst1.0[1], dst1.0[2], dst1.0[3]
        );

        if dst0.0[0] == 1.0 && dst1.0[0] == 2.0 {
            eprintln!("  → PAIR LOAD CONFIRMED: X[0]=src[0:16], X[1]=src[16:32]");
        } else if dst0.0[0] == 1.0 && dst1.0[0] == 0.0 {
            eprintln!("  → Bit 62 ignored: only X[0] loaded, X[1] unchanged");
        } else {
            eprintln!(
                "  → UNEXPECTED: X[0][0]={}, X[1][0]={}",
                dst0.0[0], dst1.0[0]
            );
        }

        // Test 3: LDX with bit 62 into X[2..3] (register 2 + pair)
        dst0.0.fill(0.0);
        dst1.0.fill(0.0);
        let pair_addr2 = (src_ptr as u64) | (1u64 << 62) | (2u64 << 56);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_LDX }>(pair_addr2);
        // Store X[2] and X[3]
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STX }>(
            (dst0.0.as_mut_ptr() as u64) | (2u64 << 56),
        );
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STX }>(
            (dst1.0.as_mut_ptr() as u64) | (3u64 << 56),
        );
        eprintln!("Pair LDX (bit62) into X[2,3]:");
        eprintln!(
            "  X[2] first 4: [{}, {}, {}, {}]",
            dst0.0[0], dst0.0[1], dst0.0[2], dst0.0[3]
        );
        eprintln!(
            "  X[3] first 4: [{}, {}, {}, {}]",
            dst1.0[0], dst1.0[1], dst1.0[2], dst1.0[3]
        );

        // Test 4: Same for LDY
        dst0.0.fill(0.0);
        dst1.0.fill(0.0);
        let pair_addr_y = (src_ptr as u64) | (1u64 << 62);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_LDY }>(pair_addr_y);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STY }>(dst0.0.as_mut_ptr() as u64);
        acpu::matrix::asm::amx_op::<{ acpu::matrix::asm::OP_STY }>(
            (dst1.0.as_mut_ptr() as u64) | (1u64 << 56),
        );
        eprintln!("Pair LDY (bit62):");
        eprintln!(
            "  Y[0] first 4: [{}, {}, {}, {}]",
            dst0.0[0], dst0.0[1], dst0.0[2], dst0.0[3]
        );
        eprintln!(
            "  Y[1] first 4: [{}, {}, {}, {}]",
            dst1.0[0], dst1.0[1], dst1.0[2], dst1.0[3]
        );

        acpu::matrix::asm::amx_clr();
    }
}
