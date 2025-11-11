use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nsv::{dumps, loads};

fn generate_test_data(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!("row{}_col{}", i, j))
                .collect()
        })
        .collect()
}

fn bench_loads_small(c: &mut Criterion) {
    let data = generate_test_data(100, 10);
    let nsv = dumps(&data);

    c.bench_function("loads_100_rows", |b| {
        b.iter(|| loads(black_box(&nsv)))
    });
}

fn bench_loads_medium(c: &mut Criterion) {
    let data = generate_test_data(1_000, 10);
    let nsv = dumps(&data);

    c.bench_function("loads_1k_rows", |b| {
        b.iter(|| loads(black_box(&nsv)))
    });
}

fn bench_loads_large(c: &mut Criterion) {
    let data = generate_test_data(10_000, 10);
    let nsv = dumps(&data);

    c.bench_function("loads_10k_rows", |b| {
        b.iter(|| loads(black_box(&nsv)))
    });
}

fn bench_loads_xlarge(c: &mut Criterion) {
    let data = generate_test_data(100_000, 10);
    let nsv = dumps(&data);

    c.bench_function("loads_100k_rows", |b| {
        b.iter(|| loads(black_box(&nsv)))
    });
}

fn bench_loads_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("loads_scaling");

    for size in [100, 500, 1_000, 5_000, 10_000, 50_000].iter() {
        let data = generate_test_data(*size, 10);
        let nsv = dumps(&data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| loads(black_box(&nsv)))
        });
    }

    group.finish();
}

fn bench_dumps(c: &mut Criterion) {
    let data = generate_test_data(10_000, 10);

    c.bench_function("dumps_10k_rows", |b| {
        b.iter(|| dumps(black_box(&data)))
    });
}

criterion_group!(
    benches,
    bench_loads_small,
    bench_loads_medium,
    bench_loads_large,
    bench_loads_xlarge,
    bench_loads_various_sizes,
    bench_dumps
);
criterion_main!(benches);
