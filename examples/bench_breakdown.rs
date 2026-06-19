use std::time::Instant;

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

fn make_nsv_near_size(target_bytes: usize) -> Vec<u8> {
    let cols = 8;
    let mut rows = 1;
    loop {
        let data = generate_mixed_data(rows, cols);
        let nsv = nsv::encode(&data);
        if nsv.len() >= target_bytes {
            return nsv.into_bytes();
        }
        rows = (rows as f64 * (target_bytes as f64 / nsv.len() as f64).max(1.1)) as usize;
    }
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_n(n: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..n / 4 {
        f();
    }
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    median(&mut times)
}

fn main() {
    #[cfg(feature = "parallel")]
    println!("parallel feature: ON (rayon threads: {})", rayon::current_num_threads());
    #[cfg(not(feature = "parallel"))]
    println!("parallel feature: OFF (sequential only)");

    println!();
    println!("{:>8}  {:>10}  {:>10}", "size", "decode_bytes", "µs/KB");
    println!("{:>8}  {:>10}  {:>10}", "----", "----------", "-----");

    for &target in &[
        8 * 1024,
        16 * 1024,
        32 * 1024,
        48 * 1024,
        64 * 1024,
        96 * 1024,
        128 * 1024,
        192 * 1024,
        256 * 1024,
        384 * 1024,
        512 * 1024,
        768 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
    ] {
        let input = make_nsv_near_size(target);
        let size = input.len();
        let t = bench_n(200, || {
            let _ = std::hint::black_box(nsv::decode_bytes(std::hint::black_box(&input)));
        });
        let kb = size as f64 / 1024.0;
        println!("{:>7.0}KB  {:>9.1}µs  {:>9.2}", kb, t, t / kb);
    }
}
