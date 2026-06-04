//! Dump ANE compiled bytecode for reverse engineering.
//!
//! Compiles a small matmul, lists all files written to the temp directory
//! by ANECompiler, and hex-dumps each file.  Use this output to locate
//! the output tensor dtype field so we can patch fp16 → fp32/int32.
//!
//! Run: cargo run -p rane --example bytecode_dump --release

use rane::mil;
use rane::Program;
use std::fmt::Write as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the smallest valid matmul.
    // ic/oc/seq = 8: smallest dimensions that align to ANE tile boundaries.
    let ic = 8usize;
    let oc = 8usize;
    let seq = 8usize;
    let program = mil::matmul(ic, oc, seq);

    println!("Compiling matmul({ic}×{oc}, seq={seq})...");
    let model = Program::compile(&program, &[])?;
    println!("  tmp_dir: {}", model.tmp_dir().display());
    println!();

    // Walk ALL files in tmp_dir recursively.
    dump_dir(model.tmp_dir(), 0);

    // Keep model alive until we've dumped everything (Drop deletes tmp_dir).
    drop(model);
    Ok(())
}

fn dump_dir(dir: &std::path::Path, depth: usize) {
    let indent = "  ".repeat(depth);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let meta = entry.metadata().unwrap();
        if meta.is_dir() {
            println!("{indent}[dir]  {name}/");
            dump_dir(&path, depth + 1);
        } else {
            let size = meta.len();
            println!("{indent}[file] {name}  ({size} bytes)");
            match std::fs::read(&path) {
                Ok(data) => {
                    // Print first 512 bytes as hex + ASCII side-by-side
                    hex_dump(&data, (size.min(512)) as usize, depth + 1);
                    if size > 512 {
                        // Also search for known dtype patterns in the full file
                        search_dtype_candidates(&data, depth + 1);
                    }
                }
                Err(e) => println!("{indent}  <read error: {e}>"),
            }
        }
    }
}

fn hex_dump(data: &[u8], max: usize, depth: usize) {
    let indent = "  ".repeat(depth);
    let n = data.len().min(max);
    for row in 0..(n + 15) / 16 {
        let lo = row * 16;
        let hi = (lo + 16).min(n);
        let mut hex = String::new();
        let mut asc = String::new();
        for &b in &data[lo..hi] {
            let _ = write!(hex, "{b:02x} ");
            asc.push(if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            });
        }
        println!("{indent}{lo:06x}  {hex:<48}  {asc}");
    }
    if data.len() > max {
        println!(
            "{indent}  ... ({} bytes total, showing first {max})",
            data.len()
        );
    }
}

/// Scan the full file for likely dtype fields.
/// Looks for the byte sequence 0x01 (fp16 dtype marker from the blob header)
/// surrounded by recognizable context.
fn search_dtype_candidates(data: &[u8], depth: usize) {
    let indent = "  ".repeat(depth);
    println!("{indent}--- dtype candidate scan (looking for 0x01/0x02/0x03 at likely offsets) ---");

    // Look for the DEADBEEF magic we know from pack_weights
    let magic = [0xEFu8, 0xBE, 0xAD, 0xDE];
    let mut found = false;
    for i in 0..data.len().saturating_sub(8) {
        if data[i..i + 4] == magic {
            let ctx_start = i.saturating_sub(16);
            let ctx_end = (i + 32).min(data.len());
            println!("{indent}  DEADBEEF @ 0x{i:x}:");
            hex_dump(&data[ctx_start..ctx_end], 64, depth + 2);
            found = true;
        }
    }
    if !found {
        println!("{indent}  no DEADBEEF magic found");
    }

    // Look for fp32 magic (0x00 0x00 0x80 0x3F = 1.0f32 little-endian) and
    // nearby dtype markers — helps locate tensor descriptors
    let f32_one = 1.0f32.to_bits().to_le_bytes();
    let mut f32_count = 0usize;
    for i in 0..data.len().saturating_sub(4) {
        if data[i..i + 4] == f32_one {
            let ctx_start = i.saturating_sub(8);
            let ctx_end = (i + 12).min(data.len());
            if f32_count < 5 {
                println!(
                    "{indent}  f32(1.0) @ 0x{i:x}:  {:?}",
                    &data[ctx_start..ctx_end]
                );
            }
            f32_count += 1;
        }
    }
    if f32_count > 0 {
        println!("{indent}  total f32(1.0) occurrences: {f32_count}");
    }

    // Show the last 64 bytes — output tensor descriptor likely lives near end
    if data.len() > 64 {
        println!("{indent}--- last 64 bytes ---");
        hex_dump(&data[data.len() - 64..], 64, depth + 1);
    }
}
