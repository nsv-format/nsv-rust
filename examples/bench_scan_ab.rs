// Controlled A/B of the two newline-scanning strategies, identical except
// for the scan primitive. Both reproduce the library's decode logic exactly;
// the ONLY difference is byte-by-byte iteration vs memchr. This isolates the
// scanning decision across a range of cell widths (the variable memchr is
// sensitive to: short cells = short haystacks = more per-call overhead).

use std::borrow::Cow;
use std::time::Instant;

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

fn decode_byteloop<'a>(input: &'a [u8]) -> Vec<Vec<Cow<'a, [u8]>>> {
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

fn decode_memchr<'a>(input: &'a [u8]) -> Vec<Vec<Cow<'a, [u8]>>> {
    let mut data = Vec::new();
    let mut row: Vec<Cow<'a, [u8]>> = Vec::new();
    let mut start = 0;
    while let Some(offset) = memchr::memchr(b'\n', &input[start..]) {
        let pos = start + offset;
        if pos > start {
            row.push(unescape_bytes(&input[start..pos]));
        } else {
            data.push(row);
            row = Vec::new();
        }
        start = pos + 1;
    }
    if start < input.len() {
        row.push(unescape_bytes(&input[start..]));
    }
    if !row.is_empty() {
        data.push(row);
    }
    data
}

// Generate NSV with a fixed cell width, no escapes (pure scan cost).
fn make_nsv_cellwidth(cell_width: usize, target_bytes: usize) -> Vec<u8> {
    let cols = 8;
    let cell: String = "x".repeat(cell_width);
    let mut out = Vec::with_capacity(target_bytes + 1024);
    while out.len() < target_bytes {
        for _ in 0..cols {
            out.extend_from_slice(cell.as_bytes());
            out.push(b'\n');
        }
        out.push(b'\n');
    }
    out
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

fn main() {
    let size = 1024 * 1024; // 1MB, sequential range
    println!("1MB input, no escapes, varying cell width (µs/KB)\n");
    println!("{:>10}  {:>10}  {:>10}  {:>8}", "cell_width", "byteloop", "memchr", "speedup");
    println!("{:>10}  {:>10}  {:>10}  {:>8}", "----------", "--------", "------", "-------");

    for &w in &[1usize, 2, 4, 8, 16, 32, 64, 128] {
        let input = make_nsv_cellwidth(w, size);
        let kb = input.len() as f64 / 1024.0;

        let t_loop = bench_n(200, || { let _ = std::hint::black_box(decode_byteloop(std::hint::black_box(&input))); });
        let t_mc = bench_n(200, || { let _ = std::hint::black_box(decode_memchr(std::hint::black_box(&input))); });

        println!("{:>10}  {:>10.3}  {:>10.3}  {:>7.2}x", w, t_loop / kb, t_mc / kb, t_loop / t_mc);
    }
}
