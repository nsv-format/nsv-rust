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

// Experimental single-pass parser WITH parallelism
fn loads_single_pass(s: &str) -> Vec<Vec<String>> {
    use memchr::memchr_iter;

    if s.is_empty() {
        return Vec::new();
    }

    let bytes = s.as_bytes();

    // Find all newline positions with SIMD - this is the ONLY scan we need
    let newlines: Vec<usize> = memchr_iter(b'\n', bytes).collect();

    if newlines.is_empty() {
        // Single cell, no newlines
        return vec![vec![unescape_single_pass(s)]];
    }

    // Identify row boundaries (where next newline is consecutive)
    let mut row_boundaries = Vec::new();
    for i in 0..newlines.len() {
        if i + 1 < newlines.len() && newlines[i + 1] == newlines[i] + 1 {
            // Found \n\n at newlines[i]
            row_boundaries.push(i); // Store index in newlines array
        }
    }

    if row_boundaries.is_empty() {
        // Single row, all newlines are cell boundaries
        return vec![parse_row_with_positions(s, &newlines, 0, newlines.len())];
    }

    // Build row ranges in the newlines array
    let mut row_ranges = Vec::new();
    let mut start = 0;

    for &boundary_idx in &row_boundaries {
        row_ranges.push((start, boundary_idx));
        start = boundary_idx + 2; // Skip past both newlines of \n\n
    }

    // Handle last row
    if start < newlines.len() {
        row_ranges.push((start, newlines.len()));
    }

    // Parse rows in parallel using pre-found newline positions
    row_ranges
        .par_iter()
        .map(|&(start_idx, end_idx)| {
            let row_start = if start_idx == 0 { 0 } else { newlines[start_idx - 1] + 1 };
            let row_end = if end_idx == 0 { 0 } else { newlines[end_idx - 1] };

            parse_row_with_positions(s, &newlines, start_idx, end_idx)
        })
        .collect()
}

// Parse a row given its cell boundary positions
fn parse_row_with_positions(s: &str, all_newlines: &[usize], start_idx: usize, end_idx: usize) -> Vec<String> {
    if start_idx >= end_idx {
        // No cell boundaries in this row
        let row_start = if start_idx == 0 { 0 } else { all_newlines[start_idx - 1] + 1 };
        let row_end = s.len();
        if row_start < row_end {
            return vec![unescape_single_pass(&s[row_start..row_end])];
        }
        return Vec::new();
    }

    let mut cells = Vec::new();
    let row_start = if start_idx == 0 { 0 } else { all_newlines[start_idx - 1] + 1 };

    // First cell
    cells.push(unescape_single_pass(&s[row_start..all_newlines[start_idx]]));

    // Middle cells
    for i in start_idx..end_idx - 1 {
        let cell_start = all_newlines[i] + 1;
        let cell_end = all_newlines[i + 1];
        if cell_start <= cell_end {
            cells.push(unescape_single_pass(&s[cell_start..cell_end]));
        }
    }

    // Last cell (if not at row boundary)
    let last_nl = all_newlines[end_idx - 1];
    if end_idx < all_newlines.len() && all_newlines[end_idx] == last_nl + 1 {
        // This is a row boundary, we're done
    } else {
        // There's content after last newline in this row
        let cell_start = last_nl + 1;
        let row_end = if end_idx < all_newlines.len() {
            all_newlines[end_idx]
        } else {
            s.len()
        };
        if cell_start < row_end {
            cells.push(unescape_single_pass(&s[cell_start..row_end]));
        }
    }

    cells
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
