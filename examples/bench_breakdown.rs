use nsv::decode_bytes;
use std::time::Instant;
use std::borrow::Cow;
use memchr::memmem;

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

fn unescape_bytes(s: &[u8]) -> Cow<'_, [u8]> {
    if s == b"\\" {
        return Cow::Owned(Vec::new());
    }
    if !s.contains(&b'\\') {
        return Cow::Borrowed(s);
    }
    let mut out = Vec::with_capacity(s.len());
    let mut escaped = false;
    for &b in s {
        if escaped {
            match b {
                b'n' => out.push(b'\n'),
                b'\\' => out.push(b'\\'),
                _ => { out.push(b'\\'); out.push(b); }
            }
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else {
            out.push(b);
        }
    }
    Cow::Owned(out)
}

fn decode_seq<'a>(input: &'a [u8]) -> Vec<Vec<Cow<'a, [u8]>>> {
    let mut data = Vec::new();
    let mut row: Vec<Cow<'a, [u8]>> = Vec::new();
    let mut start = 0;
    for (pos, &b) in input.iter().enumerate() {
        if b == b'\n' {
            if pos > start {
                row.push(unescape_bytes(&input[start..pos]));
            } else {
                data.push(row);
                row = Vec::new();
            }
            start = pos + 1;
        }
    }
    if start < input.len() {
        row.push(unescape_bytes(&input[start..]));
    }
    if !row.is_empty() {
        data.push(row);
    }
    data
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_n(n: usize, mut f: impl FnMut()) -> f64 {
    // warmup
    for _ in 0..n/4 { f(); }
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos() as f64 / 1000.0); // µs
    }
    median(&mut times)
}

fn main() {
    let num_threads = rayon::current_num_threads();
    println!("rayon threads: {}", num_threads);
    println!();

    for &target in &[64*1024, 128*1024, 256*1024, 512*1024] {
        let input = make_nsv_near_size(target);
        let size = input.len();
        println!("=== {}KB ({} bytes) ===", size / 1024, size);

        // 1. Full sequential decode
        let t_seq = bench_n(200, || { let _ = std::hint::black_box(decode_seq(&input)); });
        println!("  sequential total:       {:.1}µs", t_seq);

        // 2. Split point finding only
        let chunk_size = size / num_threads;
        let t_split = bench_n(2000, || {
            let finder = memmem::Finder::new(b"\n\n");
            let mut splits = Vec::with_capacity(num_threads + 1);
            splits.push(0usize);
            for i in 1..num_threads {
                let nominal = i * chunk_size;
                if let Some(offset) = finder.find(&input[nominal..]) {
                    let split = nominal + offset + 2;
                    if split < size { splits.push(split); }
                }
            }
            splits.push(size);
            splits.dedup();
            std::hint::black_box(splits);
        });
        println!("  split point scan:       {:.1}µs", t_split);

        // 3. Sequential parse of each chunk (serially, to measure pure parse cost without rayon)
        let finder = memmem::Finder::new(b"\n\n");
        let mut splits = Vec::with_capacity(num_threads + 1);
        splits.push(0usize);
        for i in 1..num_threads {
            let nominal = i * chunk_size;
            if let Some(offset) = finder.find(&input[nominal..]) {
                let split = nominal + offset + 2;
                if split < size { splits.push(split); }
            }
        }
        splits.push(size);
        splits.dedup();
        let chunks: Vec<&[u8]> = splits.windows(2).map(|w| &input[w[0]..w[1]]).collect();

        let t_chunks_serial = bench_n(200, || {
            let results: Vec<_> = chunks.iter().map(|c| decode_seq(c)).collect();
            std::hint::black_box(results);
        });
        println!("  chunks parsed serially: {:.1}µs ({} chunks)", t_chunks_serial, chunks.len());

        // 4. Chunks parsed with rayon par_iter
        let t_chunks_par = bench_n(200, || {
            use rayon::prelude::*;
            let results: Vec<_> = chunks.par_iter().map(|c| decode_seq(c)).collect();
            std::hint::black_box(results);
        });
        println!("  chunks parsed rayon:    {:.1}µs", t_chunks_par);

        // 5. Concatenation cost only (parse once, measure merge)
        let pre_parsed: Vec<Vec<Vec<Cow<[u8]>>>> = chunks.iter().map(|c| decode_seq(c)).collect();
        let t_concat = bench_n(2000, || {
            let total_rows: usize = pre_parsed.iter().map(|r| r.len()).sum();
            let mut result = Vec::with_capacity(total_rows);
            for chunk_rows in &pre_parsed {
                result.extend(chunk_rows.iter().map(|row| row.clone()));
            }
            std::hint::black_box(result);
        });
        println!("  concatenation:          {:.1}µs", t_concat);

        // 6. Full parallel path via library
        let t_par = bench_n(200, || { let _ = std::hint::black_box(decode_bytes(&input)); });
        println!("  library decode_bytes:   {:.1}µs", t_par);

        println!("  par/seq ratio:          {:.2}x", t_seq / t_par);
        println!();
    }
}
