use std::borrow::Cow;
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
    for _ in 0..n / 4 { f(); }
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    median(&mut times)
}

// ── Arena decode ─────────────────────────────────────────────────────
//
// Cells are either slices of the input (no unescaping) or slices of a
// bump-allocated arena (unescaped). One big allocation instead of
// per-cell mallocs.

#[derive(Clone, Copy)]
enum CellRef {
    Input(u32, u32),
    Arena(u32, u32),
}

struct ArenaRows {
    arena: Vec<u8>,
    row_offsets: Vec<u32>,
    cells: Vec<CellRef>,
}

fn unescape_into(s: &[u8], arena: &mut Vec<u8>) {
    if s == b"\\" {
        return;
    }
    let mut escaped = false;
    for &b in s {
        if escaped {
            match b {
                b'n' => arena.push(b'\n'),
                b'\\' => arena.push(b'\\'),
                _ => { arena.push(b'\\'); arena.push(b); }
            }
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else {
            arena.push(b);
        }
    }
}

fn decode_arena(input: &[u8]) -> ArenaRows {
    let mut arena = Vec::with_capacity(input.len() / 4);
    let mut row_offsets: Vec<u32> = Vec::new();
    let mut cells: Vec<CellRef> = Vec::new();
    let mut start = 0;
    let mut row_start: u32 = 0;

    while let Some(offset) = memchr::memchr(b'\n', &input[start..]) {
        let pos = start + offset;
        if pos > start {
            let cell = &input[start..pos];
            if !cell.contains(&b'\\') {
                cells.push(CellRef::Input(start as u32, pos as u32));
            } else {
                let arena_start = arena.len() as u32;
                unescape_into(cell, &mut arena);
                cells.push(CellRef::Arena(arena_start, arena.len() as u32));
            }
        } else {
            row_offsets.push(row_start);
            row_start = cells.len() as u32;
        }
        start = pos + 1;
    }

    if start < input.len() {
        let cell = &input[start..];
        if !cell.contains(&b'\\') {
            cells.push(CellRef::Input(start as u32, input.len() as u32));
        } else {
            let arena_start = arena.len() as u32;
            unescape_into(cell, &mut arena);
            cells.push(CellRef::Arena(arena_start, arena.len() as u32));
        }
    }

    if cells.len() as u32 > row_start {
        row_offsets.push(row_start);
    }

    ArenaRows { arena, row_offsets, cells }
}

fn arena_to_cow<'a>(ar: &ArenaRows, input: &'a [u8]) -> Vec<Vec<Cow<'a, [u8]>>> {
    let num_rows = ar.row_offsets.len();
    let mut result = Vec::with_capacity(num_rows);

    for (i, &offset) in ar.row_offsets.iter().enumerate() {
        let start = offset as usize;
        let end = if i + 1 < num_rows { ar.row_offsets[i + 1] as usize } else { ar.cells.len() };
        let mut row = Vec::with_capacity(end - start);
        for &cell in &ar.cells[start..end] {
            match cell {
                CellRef::Input(s, e) => row.push(Cow::Borrowed(&input[s as usize..e as usize])),
                CellRef::Arena(s, e) => row.push(Cow::Owned(ar.arena[s as usize..e as usize].to_vec())),
            }
        }
        result.push(row);
    }
    result
}

// ── Two-pass decode ──────────────────────────────────────────────────
//
// Pass 1: scan for cell/row boundaries (offsets only, no data work).
// Pass 2: unescape each cell.

struct CellBoundary {
    start: u32,
    end: u32,
}

struct RowIndex {
    boundaries: Vec<CellBoundary>,
    row_offsets: Vec<u32>,
}

fn pass1_scan(input: &[u8]) -> RowIndex {
    let mut boundaries = Vec::new();
    let mut row_offsets: Vec<u32> = Vec::new();
    let mut start = 0;
    let mut row_start: u32 = 0;

    while let Some(offset) = memchr::memchr(b'\n', &input[start..]) {
        let pos = start + offset;
        if pos > start {
            boundaries.push(CellBoundary { start: start as u32, end: pos as u32 });
        } else {
            row_offsets.push(row_start);
            row_start = boundaries.len() as u32;
        }
        start = pos + 1;
    }

    if start < input.len() {
        boundaries.push(CellBoundary { start: start as u32, end: input.len() as u32 });
    }

    if boundaries.len() as u32 > row_start {
        row_offsets.push(row_start);
    }

    RowIndex { boundaries, row_offsets }
}

fn pass2_unescape<'a>(input: &'a [u8], idx: &RowIndex) -> Vec<Vec<Cow<'a, [u8]>>> {
    let num_rows = idx.row_offsets.len();
    let mut result = Vec::with_capacity(num_rows);

    for (i, &offset) in idx.row_offsets.iter().enumerate() {
        let start = offset as usize;
        let end = if i + 1 < num_rows { idx.row_offsets[i + 1] as usize } else { idx.boundaries.len() };
        let mut row = Vec::with_capacity(end - start);
        for b in &idx.boundaries[start..end] {
            row.push(nsv::unescape_bytes(&input[b.start as usize..b.end as usize]));
        }
        result.push(row);
    }
    result
}

fn decode_two_pass<'a>(input: &'a [u8]) -> Vec<Vec<Cow<'a, [u8]>>> {
    let idx = pass1_scan(input);
    pass2_unescape(input, &idx)
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    println!("{:>8}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "size", "current", "arena", "arena+conv", "two-pass", "2p-pass1");
    println!("{:>8}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "----", "µs/KB", "µs/KB", "µs/KB", "µs/KB", "µs/KB");

    for &target in &[
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
    ] {
        let input = make_nsv_near_size(target);
        let size = input.len();
        let kb = size as f64 / 1024.0;
        let iters = if size > 16 * 1024 * 1024 { 20 } else if size > 4 * 1024 * 1024 { 40 } else { 100 };

        let t_current = bench_n(iters, || {
            let _ = std::hint::black_box(nsv::decode_bytes(std::hint::black_box(&input)));
        });

        let t_arena = bench_n(iters, || {
            let _ = std::hint::black_box(decode_arena(std::hint::black_box(&input)));
        });

        let t_arena_conv = bench_n(iters, || {
            let ar = decode_arena(std::hint::black_box(&input));
            let _ = std::hint::black_box(arena_to_cow(&ar, &input));
        });

        let t_two_pass = bench_n(iters, || {
            let _ = std::hint::black_box(decode_two_pass(std::hint::black_box(&input)));
        });

        let t_pass1 = bench_n(iters, || {
            let _ = std::hint::black_box(pass1_scan(std::hint::black_box(&input)));
        });

        if size >= 1024 * 1024 {
            println!("{:>6.0}MB  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}",
                size as f64 / (1024.0 * 1024.0),
                t_current / kb, t_arena / kb, t_arena_conv / kb, t_two_pass / kb, t_pass1 / kb);
        } else {
            println!("{:>5.0}KB  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}",
                kb, t_current / kb, t_arena / kb, t_arena_conv / kb, t_two_pass / kb, t_pass1 / kb);
        }
    }
}
