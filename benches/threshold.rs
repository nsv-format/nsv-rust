use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nsv::{encode, decode_bytes};

fn generate_mixed_data(rows: usize, cols: usize) -> Vec<Vec<String>> {
    let patterns: &[&str] = &[
        "John Smith",
        "john.smith@example.com",
        "2024-03-15",
        "1234.56",
        r"C:\Users\Documents\file.txt",
        "Line one\nLine two\nLine three",
        "",
        "Normal cell with no special chars at all",
        r#"{"key": "value", "n": 42}"#,
        "Springfield, IL 62704",
    ];
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    let idx = (i * cols + j) % patterns.len();
                    if patterns[idx].is_empty() {
                        String::new()
                    } else {
                        format!("{} [{}]", patterns[idx], i)
                    }
                })
                .collect()
        })
        .collect()
}

fn make_nsv_near_size(target_bytes: usize) -> (Vec<u8>, usize, usize) {
    let cols = 8;
    let mut rows = 1;
    loop {
        let data = generate_mixed_data(rows, cols);
        let nsv = encode(&data);
        let size = nsv.len();
        if size >= target_bytes {
            return (nsv.into_bytes(), rows, size);
        }
        rows = (rows as f64 * (target_bytes as f64 / size as f64).max(1.1)) as usize;
    }
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold");
    group.sample_size(50);

    let targets = [
        8 * 1024,
        16 * 1024,
        32 * 1024,
        48 * 1024,
        64 * 1024,
        96 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
    ];

    println!();
    for &target in &targets {
        let (nsv_bytes, rows, actual_size) = make_nsv_near_size(target);
        let label = if actual_size >= 1024 * 1024 {
            format!("{:.0}MB_{}r", actual_size as f64 / (1024.0 * 1024.0), rows)
        } else {
            format!("{}KB_{}r", actual_size / 1024, rows)
        };
        println!("  {} → {} bytes, {} rows x 8 cols", label, actual_size, rows);

        group.bench_with_input(
            BenchmarkId::new("decode_bytes", &label),
            &nsv_bytes,
            |b, data| b.iter(|| decode_bytes(black_box(data))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
