use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nsv::{encode, decode, decode_bytes, decode_bytes_projected, decode_lazy};

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
    let nsv = encode(&data);

    c.bench_function("loads_100_rows", |b| {
        b.iter(|| decode(black_box(&nsv)))
    });
}

fn bench_loads_medium(c: &mut Criterion) {
    let data = generate_test_data(1_000, 10);
    let nsv = encode(&data);

    c.bench_function("loads_1k_rows", |b| {
        b.iter(|| decode(black_box(&nsv)))
    });
}

fn bench_loads_large(c: &mut Criterion) {
    let data = generate_test_data(10_000, 10);
    let nsv = encode(&data);

    c.bench_function("loads_10k_rows", |b| {
        b.iter(|| decode(black_box(&nsv)))
    });
}

fn bench_loads_xlarge(c: &mut Criterion) {
    let data = generate_test_data(100_000, 10);
    let nsv = encode(&data);

    c.bench_function("loads_100k_rows", |b| {
        b.iter(|| decode(black_box(&nsv)))
    });
}

fn bench_loads_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("loads_scaling");

    for size in [100, 500, 1_000, 5_000, 10_000, 50_000].iter() {
        let data = generate_test_data(*size, 10);
        let nsv = encode(&data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| decode(black_box(&nsv)))
        });
    }

    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let data = generate_test_data(10_000, 10);

    c.bench_function("dumps_10k_rows", |b| {
        b.iter(|| encode(black_box(&data)))
    });
}

// ── Projection benchmarks ────────────────────────────────────────────

fn bench_projection_10k(c: &mut Criterion) {
    let data = generate_test_data(10_000, 10);
    let nsv_str = encode(&data);
    let nsv_bytes = nsv_str.as_bytes();

    let mut group = c.benchmark_group("projection_10k_x_10");

    group.bench_function("full_decode", |b| {
        b.iter(|| decode_bytes(black_box(nsv_bytes)))
    });

    group.bench_function("lazy_index_only", |b| {
        b.iter(|| decode_lazy(black_box(nsv_bytes)))
    });

    group.bench_function("projected_1_of_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0]))
    });

    group.bench_function("projected_2_of_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 5]))
    });

    group.bench_function("projected_5_of_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 2, 4, 6, 8]))
    });

    group.bench_function("projected_all_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    });

    group.finish();
}

fn bench_projection_100k(c: &mut Criterion) {
    let data = generate_test_data(100_000, 10);
    let nsv_str = encode(&data);
    let nsv_bytes = nsv_str.as_bytes();

    let mut group = c.benchmark_group("projection_100k_x_10");

    group.bench_function("full_decode", |b| {
        b.iter(|| decode_bytes(black_box(nsv_bytes)))
    });

    group.bench_function("lazy_index_only", |b| {
        b.iter(|| decode_lazy(black_box(nsv_bytes)))
    });

    group.bench_function("projected_1_of_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0]))
    });

    group.bench_function("projected_2_of_10", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 5]))
    });

    group.finish();
}

fn bench_projection_wide(c: &mut Criterion) {
    // Wide table: 1000 rows x 100 columns — projection shines here
    let data = generate_test_data(1_000, 100);
    let nsv_str = encode(&data);
    let nsv_bytes = nsv_str.as_bytes();

    let mut group = c.benchmark_group("projection_1k_x_100");

    group.bench_function("full_decode", |b| {
        b.iter(|| decode_bytes(black_box(nsv_bytes)))
    });

    group.bench_function("projected_1_of_100", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[50]))
    });

    group.bench_function("projected_5_of_100", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 25, 50, 75, 99]))
    });

    group.bench_function("projected_10_of_100", |b| {
        b.iter(|| decode_bytes_projected(black_box(nsv_bytes), &[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_loads_small,
    bench_loads_medium,
    bench_loads_large,
    bench_loads_xlarge,
    bench_loads_various_sizes,
    bench_encode,
    bench_projection_10k,
    bench_projection_100k,
    bench_projection_wide,
);
criterion_main!(benches);
