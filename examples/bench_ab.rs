//! A/B benchmark: three parallel decoding strategies.
//!
//!   OLD     — sequential memmem(\n\n) scan of entire input, then rayon per row
//!   SERIAL  — sequential memchr(\n) scan of entire input, rayon only for unescape
//!   CHUNKED — split input into N equal byte chunks, each worker does full parse
//!
//! Usage: cargo run --release --example bench_ab

use std::time::Instant;

use memchr::memmem;
use rayon::prelude::*;

// ── shared unescape (identical in all paths) ─────────────────────────────

fn unescape_bytes(s: &[u8]) -> Vec<u8> {
    if s == b"\\" {
        return Vec::new();
    }
    if !s.contains(&b'\\') {
        return s.to_vec();
    }
    let mut out = Vec::with_capacity(s.len());
    let mut escaped = false;
    for &b in s {
        if escaped {
            match b {
                b'n' => out.push(b'\n'),
                b'\\' => out.push(b'\\'),
                _ => {
                    out.push(b'\\');
                    out.push(b);
                }
            }
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else {
            out.push(b);
        }
    }
    out
}

// ── OLD: sequential memmem(\n\n) scan + rayon per row ────────────────────

fn parse_row_bytes(row: &[u8]) -> Vec<Vec<u8>> {
    if row.is_empty() {
        return Vec::new();
    }
    let mut cells = Vec::new();
    let mut start = 0;
    for (pos, &b) in row.iter().enumerate() {
        if b == b'\n' {
            if pos > start {
                cells.push(unescape_bytes(&row[start..pos]));
            } else {
                cells.push(Vec::new());
            }
            start = pos + 1;
        }
    }
    if start < row.len() {
        cells.push(unescape_bytes(&row[start..]));
    }
    cells
}

fn decode_old(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    if input.is_empty() {
        return Vec::new();
    }

    let finder = memmem::Finder::new(b"\n\n");
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while let Some(offset) = finder.find(&input[pos..]) {
        let abs_pos = pos + offset;
        boundaries.push(abs_pos);
        let mut check_pos = abs_pos + 2;
        while check_pos < input.len() && input[check_pos] == b'\n' {
            boundaries.push(check_pos - 1);
            check_pos += 1;
        }
        pos = check_pos;
    }

    if boundaries.is_empty() {
        let row = parse_row_bytes(input);
        return if row.is_empty() {
            Vec::new()
        } else {
            vec![row]
        };
    }

    let mut row_slices: Vec<&[u8]> = Vec::new();
    let mut start = 0;
    for &boundary in &boundaries {
        if boundary < start {
            row_slices.push(b"");
            start = boundary + 2;
        } else {
            row_slices.push(&input[start..boundary]);
            start = boundary + 2;
        }
    }
    if start < input.len() {
        row_slices.push(&input[start..]);
    }

    row_slices
        .par_iter()
        .map(|&slice| parse_row_bytes(slice))
        .collect()
}

// ── SERIAL: sequential memchr(\n) scan, rayon only for unescape ──────────

fn decode_serial(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut rows: Vec<Vec<&[u8]>> = Vec::new();
    let mut current_row: Vec<&[u8]> = Vec::new();
    let mut cell_start: usize = 0;

    for pos in memchr::memchr_iter(b'\n', input) {
        if pos > cell_start {
            current_row.push(&input[cell_start..pos]);
        } else {
            rows.push(std::mem::take(&mut current_row));
        }
        cell_start = pos + 1;
    }

    if cell_start < input.len() {
        current_row.push(&input[cell_start..]);
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }

    rows.par_iter()
        .map(|row| row.iter().map(|&cell| unescape_bytes(cell)).collect())
        .collect()
}

// ── CHUNKED: uses the library's new chunked parallel implementation ──────

fn decode_chunked(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    nsv::decode_bytes(input)
}

// ── Champernowne fixture generator ──────────────────────────────────────

fn generate_champernowne(target_bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(target_bytes + 1024);
    let mut n: u64 = 1;

    while buf.len() < target_bytes {
        let s = n.to_string();
        for b in s.bytes() {
            buf.push(b);
            buf.push(b'\n');
        }
        buf.push(b'\n');
        n += 1;
    }

    eprintln!(
        "Generated Champernowne fixture: {} bytes ({:.1} MB), {} rows",
        buf.len(),
        buf.len() as f64 / (1024.0 * 1024.0),
        n - 1
    );
    buf
}

// ── Timing harness ──────────────────────────────────────────────────────

fn bench_fn(name: &str, input: &[u8], f: fn(&[u8]) -> Vec<Vec<Vec<u8>>>, iters: u32) {
    // Warm up
    let _ = f(input);

    let mut times = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        let result = f(input);
        let elapsed = t.elapsed();
        std::hint::black_box(&result);
        drop(result);
        times.push(elapsed);
    }

    times.sort();
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let mean: std::time::Duration = times.iter().sum::<std::time::Duration>() / iters;
    let throughput = input.len() as f64 / median.as_secs_f64() / (1024.0 * 1024.0);

    eprintln!(
        "  {name:>8}: median {median:>9.2?}  mean {mean:>9.2?}  min {min:>9.2?}  max {max:>9.2?}  ({throughput:.0} MB/s)"
    );
}

fn main() {
    eprintln!("Cores: {}", rayon::current_num_threads());
    let iters = 7;

    // ── Champernowne: many tiny cells, high newline density ─────────
    let target = 2 * 1024 * 1024;
    let input = generate_champernowne(target);

    eprintln!("\n=== Champernowne ~2 MB ({iters} iterations) ===");
    bench_fn("OLD", &input, decode_old, iters);
    bench_fn("SERIAL", &input, decode_serial, iters);
    bench_fn("CHUNKED", &input, decode_chunked, iters);

    // Verify correctness
    let old_result = decode_old(&input);
    let chunked_result = decode_chunked(&input);
    assert_eq!(old_result.len(), chunked_result.len(), "row count mismatch");
    assert_eq!(old_result, chunked_result, "output mismatch");
    eprintln!("  Correctness: PASS ({} rows identical)", old_result.len());
    drop(old_result);
    drop(chunked_result);

    // ── Synthetic wide table: fewer rows, more cells per row ────────
    eprintln!("\n=== Synthetic 100K rows x 10 cols ===");
    let wide_data = nsv::encode(
        &(0..100_000u64)
            .map(|i| (0..10).map(|j| format!("row{}_col{}", i, j)).collect())
            .collect::<Vec<Vec<String>>>(),
    );
    let wide_bytes = wide_data.as_bytes();
    eprintln!(
        "  Input: {} bytes ({:.1} MB)",
        wide_bytes.len(),
        wide_bytes.len() as f64 / (1024.0 * 1024.0)
    );
    bench_fn("OLD", wide_bytes, decode_old, iters);
    bench_fn("SERIAL", wide_bytes, decode_serial, iters);
    bench_fn("CHUNKED", wide_bytes, decode_chunked, iters);

    // ── Synthetic with escapes: backslash-heavy data ────────────────
    eprintln!("\n=== Escape-heavy 50K rows x 10 cols ===");
    let escape_data = nsv::encode(
        &(0..50_000u64)
            .map(|i| {
                (0..10)
                    .map(|j| {
                        if j % 3 == 0 {
                            format!("line1\nline2\nrow{}_col{}", i, j)
                        } else if j % 3 == 1 {
                            format!("path\\to\\row{}\\col{}", i, j)
                        } else {
                            format!("row{}_col{}", i, j)
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<String>>>(),
    );
    let escape_bytes = escape_data.as_bytes();
    eprintln!(
        "  Input: {} bytes ({:.1} MB)",
        escape_bytes.len(),
        escape_bytes.len() as f64 / (1024.0 * 1024.0)
    );
    bench_fn("OLD", escape_bytes, decode_old, iters);
    bench_fn("SERIAL", escape_bytes, decode_serial, iters);
    bench_fn("CHUNKED", escape_bytes, decode_chunked, iters);
}
