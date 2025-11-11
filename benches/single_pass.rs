use criterion::{black_box, criterion_group, criterion_main, Criterion};
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

// Experimental single-pass parser
fn loads_single_pass(s: &str) -> Vec<Vec<String>> {
    use memchr::memchr_iter;

    if s.is_empty() {
        return Vec::new();
    }

    let bytes = s.as_bytes();

    // Find all newline positions with SIMD
    let newlines: Vec<usize> = memchr_iter(b'\n', bytes).collect();

    if newlines.is_empty() {
        // Single cell, no newlines
        return vec![vec![unescape_single_pass(s)]];
    }

    let mut result = Vec::new();
    let mut current_row = Vec::new();
    let mut cell_start = 0;

    for i in 0..newlines.len() {
        let nl_pos = newlines[i];

        // Check if next position is also a newline (row boundary)
        let is_row_boundary = i + 1 < newlines.len() && newlines[i + 1] == nl_pos + 1;

        if is_row_boundary {
            // This is first \n of \n\n
            let cell_text = &s[cell_start..nl_pos];
            if !cell_text.is_empty() || cell_start == nl_pos {
                current_row.push(unescape_single_pass(cell_text));
            }
            result.push(current_row);
            current_row = Vec::new();
            cell_start = newlines[i + 1] + 1; // Skip past both newlines
            // Note: we'll process the second \n in next iteration but skip it
        } else if i > 0 && newlines[i - 1] == nl_pos - 1 {
            // This is second \n of \n\n, already handled
            continue;
        } else {
            // Regular cell boundary
            let cell_text = &s[cell_start..nl_pos];
            current_row.push(unescape_single_pass(cell_text));
            cell_start = nl_pos + 1;
        }
    }

    // Handle trailing data
    if cell_start < s.len() {
        let cell_text = &s[cell_start..];
        current_row.push(unescape_single_pass(cell_text));
    }

    if !current_row.is_empty() {
        result.push(current_row);
    }

    result
}

fn unescape_single_pass(s: &str) -> String {
    if s == "\\" {
        return String::new();
    }

    if !s.contains('\\') {
        return s.to_string();
    }

    let mut out = String::new();
    let mut escaped = false;

    for c in s.chars() {
        if escaped {
            match c {
                'n' => out.push('\n'),
                '\\' => out.push('\\'),
                _ => {
                    out.push('\\');
                    out.push(c);
                }
            }
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else {
            out.push(c);
        }
    }

    out
}

fn bench_single_pass_vs_current(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pass_comparison");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let data = generate_test_data(*size, 10);
        let nsv_str = dumps(&data);

        group.bench_function(&format!("current_{}", size), |b| {
            b.iter(|| loads(black_box(&nsv_str)))
        });

        group.bench_function(&format!("single_pass_{}", size), |b| {
            b.iter(|| loads_single_pass(black_box(&nsv_str)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_pass_vs_current);
criterion_main!(benches);
