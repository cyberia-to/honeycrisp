//! Phase 1 smoke test for the SME / streaming-mode driver.
//!
//! Enters streaming mode, reads SVL via RDSVL, prints it, leaves
//! streaming mode. On a non-M4 host prints SKIP and exits zero.

fn main() {
    let caps = acpu::scan();
    println!("chip: {}", caps.chip);

    if !caps.has_sme {
        println!("SKIP: FEAT_SME not present (need M4 or later)");
        return;
    }

    println!("FEAT_SME:       yes");
    println!("FEAT_SME2:      {}", caps.has_sme2);
    println!("FEAT_F64F64:    {}", caps.has_sme_f64f64);
    println!("FEAT_I16I64:    {}", caps.has_sme_i16i64);
    println!("probed SVL:     {} B", caps.svl_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        let stream = acpu::Stream::new().expect("Stream::new failed despite has_sme");
        let svl_runtime = stream.svl_bytes();
        println!("runtime SVL:    {svl_runtime} B (RDSVL)");
        println!("f32 lanes/vec:  {}", stream.svl_lanes_f32());
        println!("f64 lanes/vec:  {}", stream.svl_lanes_f64());
        assert_eq!(
            svl_runtime as u16, caps.svl_bytes,
            "RDSVL must agree with probed SVL"
        );

        // Re-zero ZA — purely exercising the encoding.
        stream.zero_za();
        println!("ZA zero-fill:   ok");

        drop(stream);
        println!("smstop:         ok");
    }
}
