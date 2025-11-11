use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use csv::{ReaderBuilder, WriterBuilder};
use nsv::{dumps, loads};
use std::io::Write;

fn generate_test_data(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!("row{}_col{}", i, j))
                .collect()
        })
        .collect()
}

fn generate_test_data_with_multiline(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    if j % 3 == 0 {
                        format!("Line1\nLine2\nrow{}_col{}", i, j)
                    } else {
                        format!("row{}_col{}", i, j)
                    }
                })
                .collect()
        })
        .collect()
}

fn data_to_csv(data: &[Vec<String>]) -> String {
    let mut wtr = WriterBuilder::new().from_writer(vec![]);
    for row in data {
        wtr.write_record(row).unwrap();
    }
    String::from_utf8(wtr.into_inner().unwrap()).unwrap()
}

fn csv_to_data(csv: &str) -> Vec<Vec<String>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(csv.as_bytes());
    rdr.records()
        .map(|r| r.unwrap().iter().map(|s| s.to_string()).collect())
        .collect()
}

fn bench_comparison_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_simple_10k");

    let data = generate_test_data(10_000, 10);
    let nsv_str = dumps(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Simple Data (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| loads(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_large_100k");

    let data = generate_test_data(100_000, 10);
    let nsv_str = dumps(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Large Data (100K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| loads(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_multiline(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_multiline_10k");

    let data = generate_test_data_with_multiline(10_000, 10);
    let nsv_str = dumps(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Multiline Data (10K rows x 10 cols with newlines) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| loads(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_wide_1k");

    let data = generate_test_data(1_000, 100); // 1K rows, 100 columns
    let nsv_str = dumps(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Wide Data (1K rows x 100 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| loads(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

criterion_group!(
    benches,
    bench_comparison_simple,
    bench_comparison_large,
    bench_comparison_multiline,
    bench_comparison_wide
);
criterion_main!(benches);
