use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use unimem::Tape;

const SIZE: usize = 64 * 1024 * 1024;

fn write_bench(ptr: *mut u8, size: usize) {
    unsafe {
        let p = ptr as *mut u64;
        let count = size / 8;
        for i in 0..count {
            std::ptr::write_volatile(p.add(i), i as u64);
        }
    }
}

fn read_bench(ptr: *const u8, size: usize) -> u64 {
    unsafe {
        let p = ptr as *const u64;
        let count = size / 8;
        let mut sum: u64 = 0;
        for i in 0..count {
            sum = sum.wrapping_add(std::ptr::read_volatile(p.add(i)));
        }
        sum
    }
}

fn prefill(ptr: *mut u8, size: usize) {
    unsafe {
        let p = ptr as *mut u64;
        for i in 0..size / 8 {
            std::ptr::write_volatile(p.add(i), i as u64);
        }
    }
}

fn bench_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandwidth");
    group.throughput(Throughput::Bytes(SIZE as u64));

    let tape = Tape::start_warm(SIZE).unwrap();
    let cyb_ptr = tape.take(SIZE, 64).unwrap();
    prefill(cyb_ptr, SIZE);

    group.bench_function("tape_write", |b| {
        b.iter(|| {
            write_bench(cyb_ptr, SIZE);
            black_box(());
        });
    });
    group.bench_function("tape_read", |b| {
        b.iter(|| {
            black_box(read_bench(cyb_ptr, SIZE));
        });
    });

    let mut vec_buf: Vec<u8> = vec![0u8; SIZE];
    let vec_ptr = vec_buf.as_mut_ptr();
    prefill(vec_ptr, SIZE);

    group.bench_function("vec_write", |b| {
        b.iter(|| {
            write_bench(vec_ptr, SIZE);
            black_box(());
        });
    });
    group.bench_function("vec_read", |b| {
        b.iter(|| {
            black_box(read_bench(vec_ptr, SIZE));
        });
    });

    let malloc_ptr = unsafe { libc::malloc(SIZE) as *mut u8 };
    prefill(malloc_ptr, SIZE);

    group.bench_function("malloc_write", |b| {
        b.iter(|| {
            write_bench(malloc_ptr, SIZE);
            black_box(());
        });
    });
    group.bench_function("malloc_read", |b| {
        b.iter(|| {
            black_box(read_bench(malloc_ptr, SIZE));
        });
    });

    group.finish();
    unsafe {
        libc::free(malloc_ptr as _);
    }
}

criterion_group!(benches, bench_bandwidth);
criterion_main!(benches);
