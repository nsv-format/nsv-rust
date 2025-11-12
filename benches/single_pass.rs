use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nsv::{dumps, loads};
use rayon::prelude::*;

fn generate_test_data(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!("row{}_col{}", i, j))
                .collect()
        })
        .collect()
}

// Experimental: Find all newlines AND backslashes upfront, skip unescape when possible
fn loads_single_pass(s: &str) -> Vec<Vec<String>> {
    use memchr::memchr_iter;

    if s.is_empty() {
        return Vec::new();
    }

    let bytes = s.as_bytes();

    // Two SIMD scans: find ALL newlines and ALL backslashes
    let newlines: Vec<usize> = memchr_iter(b'\n', bytes).collect();
    let backslashes: Vec<usize> = memchr_iter(b'\\', bytes).collect();

    if newlines.is_empty() {
        // Single cell
        let needs_unescape = !backslashes.is_empty();
        if needs_unescape {
            return vec![vec![unescape_single_pass(s)]];
        } else {
            return vec![vec![s.to_string()]];
        }
    }

    // Identify row boundaries
    let mut row_boundaries = Vec::new();
    for i in 0..newlines.len() - 1 {
        if newlines[i + 1] == newlines[i] + 1 {
            row_boundaries.push(i);
        }
    }

    if row_boundaries.is_empty() {
        // Single row
        return vec![parse_row_fast(s, &newlines, &backslashes, 0, newlines.len())];
    }

    // Build row ranges
    let mut row_ranges = Vec::new();
    let mut start = 0;

    for &boundary_idx in &row_boundaries {
        row_ranges.push((start, boundary_idx));
        start = boundary_idx + 2;
    }

    if start < newlines.len() {
        row_ranges.push((start, newlines.len()));
    }

    // Parse rows in parallel
    row_ranges
        .par_iter()
        .map(|&(start_idx, end_idx)| {
            parse_row_fast(s, &newlines, &backslashes, start_idx, end_idx)
        })
        .collect()
}

// Parse row with fast path for cells without backslashes
fn parse_row_fast(s: &str, newlines: &[usize], backslashes: &[usize], start_idx: usize, end_idx: usize) -> Vec<String> {
    if start_idx >= end_idx {
        // Empty row or single cell without newline
        let row_start = if start_idx == 0 { 0 } else { newlines[start_idx - 1] + 1 };
        let row_end = s.len();
        if row_start >= row_end {
            return Vec::new();
        }

        let has_backslash = has_backslash_in_range(backslashes, row_start, row_end);
        if has_backslash {
            return vec![unescape_single_pass(&s[row_start..row_end])];
        } else {
            return vec![s[row_start..row_end].to_string()];
        }
    }

    let mut cells = Vec::new();
    let row_start = if start_idx == 0 { 0 } else { newlines[start_idx - 1] + 1 };

    // Binary search to find first backslash >= row_start
    let mut bs_idx = match backslashes.binary_search(&row_start) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };

    // Process each cell
    let mut cell_start = row_start;
    for i in start_idx..end_idx {
        let cell_end = newlines[i];

        // Advance bs_idx to first backslash >= cell_start
        while bs_idx < backslashes.len() && backslashes[bs_idx] < cell_start {
            bs_idx += 1;
        }

        // Check if there's a backslash in [cell_start, cell_end)
        let has_backslash = bs_idx < backslashes.len() && backslashes[bs_idx] < cell_end;

        if has_backslash {
            cells.push(unescape_single_pass(&s[cell_start..cell_end]));
        } else {
            cells.push(s[cell_start..cell_end].to_string());
        }

        cell_start = cell_end + 1;
    }

    // Handle trailing cell
    if cell_start < s.len() {
        let cell_end = if end_idx < newlines.len() {
            newlines[end_idx]
        } else {
            s.len()
        };

        if cell_start < cell_end {
            while bs_idx < backslashes.len() && backslashes[bs_idx] < cell_start {
                bs_idx += 1;
            }
            let has_backslash = bs_idx < backslashes.len() && backslashes[bs_idx] < cell_end;

            if has_backslash {
                cells.push(unescape_single_pass(&s[cell_start..cell_end]));
            } else {
                cells.push(s[cell_start..cell_end].to_string());
            }
        }
    }

    cells
}

// Binary search to check if any backslash in range [start, end)
fn has_backslash_in_range(backslashes: &[usize], start: usize, end: usize) -> bool {
    let idx = match backslashes.binary_search(&start) {
        Ok(idx) => return true, // Exact match at start
        Err(idx) => idx,
    };
    idx < backslashes.len() && backslashes[idx] < end
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
