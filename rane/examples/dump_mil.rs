use rane::mil;
fn main() {
    let src = mil::matmul(64, 64, 64);
    eprintln!("{}", src.as_str());
}
