use criterion::{black_box, criterion_group, criterion_main, Criterion};
use unimem::Tape;

fn bench_take(c: &mut Criterion) {
    let mut group = c.benchmark_group("take_comparison");

    let tape = Tape::start_warm(1 << 30).unwrap();

    group.bench_function("tape_take_64b", |b| {
        tape.clear();
        b.iter(|| black_box(tape.take(64, 64)));
    });

    group.bench_function("tape_take_4kb", |b| {
        tape.clear();
        b.iter(|| black_box(tape.take(4096, 64)));
    });

    group.bench_function("tape_take_1mb", |b| {
        tape.clear();
        b.iter(|| black_box(tape.take(1 << 20, 64)));
    });

    group.bench_function("vec_4kb", |b| {
        b.iter(|| {
            let v = Vec::<u8>::with_capacity(4096);
            black_box(v.as_ptr());
        });
    });

    group.bench_function("malloc_4kb", |b| {
        b.iter(|| unsafe {
            let p = libc::malloc(4096);
            black_box(p);
            libc::free(p);
        });
    });

    group.bench_function("tape_clear", |b| {
        b.iter(|| {
            tape.clear();
            black_box(());
        });
    });

    group.finish();
}

fn bench_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_comparison");

    let grid: unimem::Grid<4096, 1024> = unimem::Grid::new().unwrap();

    group.bench_function("grid_take_give", |b| {
        b.iter(|| {
            let cell = grid.take().unwrap();
            black_box(cell.address());
            grid.give(cell);
        });
    });

    group.bench_function("malloc_free_4kb", |b| {
        b.iter(|| unsafe {
            let p = libc::malloc(4096);
            black_box(p);
            libc::free(p);
        });
    });

    group.finish();
}

fn bench_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_open");

    group.bench_function("block_open_16mb", |b| {
        b.iter(|| {
            let b = unimem::Block::open(16 << 20).unwrap();
            black_box(b.address());
        });
    });

    group.bench_function("mmap_16mb", |b| {
        b.iter(|| unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                16 << 20,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            );
            black_box(p);
            libc::munmap(p, 16 << 20);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_take, bench_grid, bench_block);
criterion_main!(benches);
