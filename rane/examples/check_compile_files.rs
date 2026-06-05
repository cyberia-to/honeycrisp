use rane::mil::matmul as make_matmul;
use rane::Program;
use std::mem;

fn main() {
    let src = make_matmul(16, 16, 1);
    let weights_fp16: Vec<u8> = {
        let n = 16 * 16;
        let mut v = vec![0u8; n * 2 + 128];
        v[0] = 0xEF;
        v[1] = 0xBE;
        v[2] = 0xAD;
        v[3] = 0xDE;
        v
    };
    let prog = Program::compile(&src, &[("@model_path/weights/w.bin", &weights_fp16)]).unwrap();
    println!("tmp_dir: {:?}", prog.tmp_dir());
    for entry in walkdir(prog.tmp_dir()) {
        println!("  {}", entry);
    }
    // Prevent destructor from deleting files
    mem::forget(prog);
    println!("Files preserved (no drop).");
}

fn walkdir(path: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            let meta =
                std::fs::metadata(&p).unwrap_or_else(|_| std::fs::symlink_metadata(&p).unwrap());
            if meta.is_dir() {
                out.extend(walkdir(&p));
            } else {
                out.push(format!("{}: {}B", p.display(), meta.len()));
            }
        }
    }
    out
}
