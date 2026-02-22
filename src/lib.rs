//! NSV (Newline-Separated Values) format implementation for Rust
//!
//! Fast parallel implementation using rayon and memchr.
//! See https://nsv-format.org for the specification.
//!
//! ## Parallel Parsing Strategy
//!
//! For files larger than 64KB, we use a chunked parallel approach:
//! 1. Pick N evenly-spaced byte positions (one per CPU core)
//! 2. For each, scan forward to the nearest `\n\n` row boundary — O(avg_row_len)
//! 3. Each worker independently parses its chunk (boundary scan + cell split + unescape)
//!
//! This works because literal `0x0A` bytes in NSV are always structural (never escaped),
//! so row alignment recovery from any byte position is a trivial forward scan.
//! The sequential phase is O(N), not O(input_len) — all real work is parallel.
//!
//! For smaller files, we use a sequential fast path to avoid thread overhead.

pub mod util;

use memchr::memmem;
use rayon::prelude::*;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Threshold for using parallel parsing (64KB)
const PARALLEL_THRESHOLD: usize = 64 * 1024;

/// Decode an NSV string into a seqseq.
pub fn decode(s: &str) -> Vec<Vec<String>> {
    decode_bytes(s.as_bytes())
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| {
                    // SAFETY: input was &str (valid UTF-8). NSV splitting and unescaping
                    // only operate on ASCII bytes (0x0A, 0x5C, 0x6E), which cannot split
                    // a multi-byte UTF-8 sequence. Each resulting cell is therefore valid UTF-8.
                    unsafe { String::from_utf8_unchecked(cell) }
                })
                .collect()
        })
        .collect()
}

/// Decode raw bytes into a seqseq of byte vectors.
/// No encoding assumption — works with any ASCII-compatible encoding.
pub fn decode_bytes(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    if input.is_empty() {
        return Vec::new();
    }

    // For small inputs, use sequential fast path
    if input.len() < PARALLEL_THRESHOLD {
        return decode_bytes_sequential(input);
    }

    decode_bytes_parallel(input)
}

/// Sequential implementation for small inputs (byte-level).
fn decode_bytes_sequential(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let mut data = Vec::new();
    let mut row: Vec<Vec<u8>> = Vec::new();
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

/// Chunked parallel implementation for large inputs (byte-level).
///
/// Splits the input into N equal-sized byte chunks (one per core), aligns each
/// split point to the nearest `\n\n` row boundary, and parses each chunk
/// independently. The sequential phase is O(N), not O(input_len).
fn decode_bytes_parallel(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = input.len() / num_threads;

    if chunk_size == 0 {
        return decode_bytes_sequential(input);
    }

    // Find N-1 split points at \n\n boundaries near evenly-spaced positions.
    // Cost: O(N * avg_row_len) — negligible compared to input size.
    let finder = memmem::Finder::new(b"\n\n");
    let mut splits = Vec::with_capacity(num_threads + 1);
    splits.push(0usize);

    for i in 1..num_threads {
        let nominal = i * chunk_size;
        if let Some(offset) = finder.find(&input[nominal..]) {
            let split = nominal + offset + 2; // byte after \n\n
            if split < input.len() {
                splits.push(split);
            }
        }
    }
    splits.push(input.len());
    splits.dedup();

    if splits.len() <= 2 {
        return decode_bytes_sequential(input);
    }

    // Parse each chunk in parallel. Each chunk starts at a row boundary
    // (or byte 0), so the sequential parser produces correct results per chunk.
    let chunks: Vec<&[u8]> = splits.windows(2).map(|w| &input[w[0]..w[1]]).collect();

    let chunk_results: Vec<Vec<Vec<Vec<u8>>>> = chunks
        .par_iter()
        .map(|chunk| decode_bytes_sequential(chunk))
        .collect();

    let total_rows: usize = chunk_results.iter().map(|r| r.len()).sum();
    let mut result = Vec::with_capacity(total_rows);
    for chunk_rows in chunk_results {
        result.extend(chunk_rows);
    }
    result
}

/// Unescape a single NSV cell.
pub fn unescape(s: &str) -> String {
    // SAFETY: unescape only removes/replaces ASCII bytes — preserves UTF-8 validity.
    unsafe { String::from_utf8_unchecked(unescape_bytes(s.as_bytes())) }
}

/// Unescape a single raw cell (byte-level).
///
/// Interprets `\` as the empty cell token (returns empty vec).
/// `\\` → `\`, `\n` → LF. Unrecognized sequences pass through.
/// Dangling backslash at end is stripped.
pub fn unescape_bytes(s: &[u8]) -> Vec<u8> {
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

/// Escape a single NSV cell.
pub fn escape(s: &str) -> String {
    // SAFETY: escape only inserts ASCII bytes (\, n) — preserves UTF-8 validity.
    unsafe { String::from_utf8_unchecked(escape_bytes(s.as_bytes())) }
}

/// Escape a single raw cell (byte-level).
///
/// Empty input → `\` (empty cell token).
/// `\` → `\\`, LF → `\n`.
pub fn escape_bytes(s: &[u8]) -> Vec<u8> {
    if s.is_empty() {
        return b"\\".to_vec();
    }

    if s.contains(&b'\n') || s.contains(&b'\\') {
        let mut out = Vec::with_capacity(s.len() + s.len() / 4);
        for &b in s {
            match b {
                b'\\' => {
                    out.push(b'\\');
                    out.push(b'\\');
                }
                b'\n' => {
                    out.push(b'\\');
                    out.push(b'n');
                }
                _ => out.push(b),
            }
        }
        out
    } else {
        s.to_vec()
    }
}

// ── Projected (column-selective) parsing ─────────────────────────────
//
// Exploits the NSV property that structure recovery (row/cell boundaries)
// is independent of unescaping.  For queries that only need a subset of
// columns, we scan the full byte stream to find boundaries but only
// unescape cells in the requested columns.
//
// Two levels of API:
//
// 1. `decode_bytes_projected` — single-pass scan that directly produces
//    the final `Vec<Vec<Vec<u8>>>` for projected columns.  No intermediate
//    index.  Best when the caller wants materialised values immediately.
//
// 2. `decode_lazy` / `decode_lazy_projected` — builds a structural index
//    (byte-range spans) without unescaping.  Cells are unescaped on demand.
//    `decode_lazy_projected` uses flat storage (single `Vec<CellSpan>`)
//    and only records spans for projected columns, eliminating the
//    per-row heap allocation of the full `Vec<Vec<CellSpan>>` index.

/// A byte range `[start, end)` inside the original input identifying a raw
/// (still-escaped) cell.
#[derive(Debug, Clone, Copy)]
pub struct CellSpan {
    pub start: usize,
    pub end: usize,
}

// ── Full lazy index (unchanged, needed for bind-time header/type sniffing) ──

/// A lazily-decoded NSV document.  The full structural index (row/cell
/// boundaries) is built up front, but cell values are only unescaped on
/// demand.  This enables zero-cost column projection: iterate over all
/// rows, but only unescape the cells that the query actually reads.
pub struct LazyNsv<'a> {
    input: &'a [u8],
    /// Each row is a `Vec<CellSpan>` — one span per cell in that row.
    pub rows: Vec<Vec<CellSpan>>,
}

impl<'a> LazyNsv<'a> {
    /// Consume this lazy handle and return just the structural index.
    /// Useful for FFI where the caller owns the input separately.
    pub fn into_rows(self) -> Vec<Vec<CellSpan>> {
        self.rows
    }

    /// Number of rows (including the header row, if any).
    #[inline]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Number of cells in `row`.
    #[inline]
    pub fn col_count(&self, row: usize) -> usize {
        self.rows.get(row).map_or(0, |r| r.len())
    }

    /// Unescape and return the cell at `(row, col)`.
    /// Returns `None` if out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<Vec<u8>> {
        let span = self.rows.get(row)?.get(col)?;
        Some(unescape_bytes(&self.input[span.start..span.end]))
    }

    /// Return the raw (still-escaped) bytes for the cell at `(row, col)`.
    pub fn get_raw(&self, row: usize, col: usize) -> Option<&'a [u8]> {
        let span = self.rows.get(row)?.get(col)?;
        Some(&self.input[span.start..span.end])
    }

    /// Materialise only the requested `columns` (by 0-based index).
    ///
    /// Returns one `Vec` per row.  Each inner `Vec` has exactly
    /// `columns.len()` entries, mapped in the same order as `columns`.
    /// Out-of-range column indices produce empty `Vec<u8>`.
    pub fn project(&self, columns: &[usize]) -> Vec<Vec<Vec<u8>>> {
        self.rows
            .iter()
            .map(|row| {
                columns
                    .iter()
                    .map(|&ci| {
                        row.get(ci)
                            .map(|span| unescape_bytes(&self.input[span.start..span.end]))
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Build a structural index over `input` without unescaping any cells.
///
/// Cost: one linear scan of the input (same as `decode_bytes` minus the
/// per-cell `unescape_bytes` calls).  For large inputs the indexing pass
/// is parallelised just like `decode_bytes`.
pub fn decode_lazy(input: &[u8]) -> LazyNsv<'_> {
    if input.is_empty() {
        return LazyNsv {
            input,
            rows: Vec::new(),
        };
    }

    let rows = if input.len() < PARALLEL_THRESHOLD {
        index_sequential(input)
    } else {
        index_parallel(input)
    };

    LazyNsv { input, rows }
}

/// Sequential structural indexing — mirrors `decode_bytes_sequential` but
/// records `CellSpan`s instead of calling `unescape_bytes`.
fn index_sequential(input: &[u8]) -> Vec<Vec<CellSpan>> {
    let mut data = Vec::new();
    let mut row: Vec<CellSpan> = Vec::new();
    let mut start = 0;

    for (pos, &b) in input.iter().enumerate() {
        if b == b'\n' {
            if pos > start {
                row.push(CellSpan { start, end: pos });
            } else {
                data.push(row);
                row = Vec::new();
            }
            start = pos + 1;
        }
    }

    if start < input.len() {
        row.push(CellSpan {
            start,
            end: input.len(),
        });
    }

    if !row.is_empty() {
        data.push(row);
    }

    data
}

/// Parallel structural indexing — mirrors `decode_bytes_parallel`.
fn index_parallel(input: &[u8]) -> Vec<Vec<CellSpan>> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = input.len() / num_threads;

    if chunk_size == 0 {
        return index_sequential(input);
    }

    let finder = memmem::Finder::new(b"\n\n");
    let mut splits = Vec::with_capacity(num_threads + 1);
    splits.push(0usize);

    for i in 1..num_threads {
        let nominal = i * chunk_size;
        if let Some(offset) = finder.find(&input[nominal..]) {
            let split = nominal + offset + 2;
            if split < input.len() {
                splits.push(split);
            }
        }
    }
    splits.push(input.len());
    splits.dedup();

    if splits.len() <= 2 {
        return index_sequential(input);
    }

    let chunks: Vec<(usize, &[u8])> = splits
        .windows(2)
        .map(|w| (w[0], &input[w[0]..w[1]]))
        .collect();

    let chunk_results: Vec<Vec<Vec<CellSpan>>> = chunks
        .par_iter()
        .map(|&(base_offset, chunk)| {
            // Index within the chunk, then shift spans to absolute positions
            let local = index_sequential(chunk);
            local
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|span| CellSpan {
                            start: span.start + base_offset,
                            end: span.end + base_offset,
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let total_rows: usize = chunk_results.iter().map(|r| r.len()).sum();
    let mut result = Vec::with_capacity(total_rows);
    for chunk_rows in chunk_results {
        result.extend(chunk_rows);
    }
    result
}

// ── Projected lazy index (flat storage, only projected columns) ─────

/// An empty/sentinel span used for padding when a row has fewer columns
/// than a projected index requires.
const EMPTY_SPAN: CellSpan = CellSpan { start: 0, end: 0 };

/// A lazily-decoded NSV document indexed only at projected columns.
///
/// Uses a single flat `Vec<CellSpan>` — one allocation total, not one per
/// row.  `spans[row * stride + proj_idx]` gives the byte range for the
/// `proj_idx`-th projected column in `row`.
pub struct ProjectedNsv<'a> {
    input: &'a [u8],
    /// Number of projected columns (stride of the flat array).
    pub stride: usize,
    pub num_rows: usize,
    /// Flat: `spans[row * stride + proj_col_idx]`.
    pub spans: Vec<CellSpan>,
}

impl<'a> ProjectedNsv<'a> {
    /// Unescape and return the cell at `(row, proj_col)`.
    pub fn get(&self, row: usize, proj_col: usize) -> Option<Vec<u8>> {
        if row >= self.num_rows || proj_col >= self.stride {
            return None;
        }
        let span = &self.spans[row * self.stride + proj_col];
        if span.start == span.end {
            // Empty span — either genuinely empty cell or out-of-range column
            return Some(Vec::new());
        }
        Some(unescape_bytes(&self.input[span.start..span.end]))
    }

    /// Return the raw (still-escaped) bytes at `(row, proj_col)`.
    pub fn get_raw(&self, row: usize, proj_col: usize) -> Option<&'a [u8]> {
        if row >= self.num_rows || proj_col >= self.stride {
            return None;
        }
        let span = &self.spans[row * self.stride + proj_col];
        Some(&self.input[span.start..span.end])
    }
}

/// Build a column-map: `col_map[original_col] = projected_index`.
/// Entries for non-projected columns are `usize::MAX`.
fn build_col_map(columns: &[usize]) -> (Vec<usize>, usize) {
    let max_col = columns.iter().copied().max().unwrap_or(0);
    let mut col_map = vec![usize::MAX; max_col + 1];
    for (proj_idx, &orig_col) in columns.iter().enumerate() {
        col_map[orig_col] = proj_idx;
    }
    (col_map, max_col)
}

/// Build a projected structural index: single-pass scan, flat storage.
///
/// Only records spans for columns whose 0-based index is in `columns`.
/// Each worker (parallel or sequential) starts at a row boundary, so it
/// always knows the current column count from 0.
pub fn decode_lazy_projected<'a>(input: &'a [u8], columns: &[usize]) -> ProjectedNsv<'a> {
    let stride = columns.len();
    if input.is_empty() || stride == 0 {
        return ProjectedNsv {
            input,
            stride,
            num_rows: 0,
            spans: Vec::new(),
        };
    }

    let (col_map, max_col) = build_col_map(columns);

    let (num_rows, spans) = if input.len() < PARALLEL_THRESHOLD {
        index_projected_sequential(input, &col_map, max_col, stride)
    } else {
        index_projected_parallel(input, &col_map, max_col, stride)
    };

    ProjectedNsv {
        input,
        stride,
        num_rows,
        spans,
    }
}

/// Sequential projected indexing.  Returns `(num_rows, flat_spans)`.
fn index_projected_sequential(
    input: &[u8],
    col_map: &[usize],
    max_col: usize,
    stride: usize,
) -> (usize, Vec<CellSpan>) {
    let mut spans = Vec::new();
    let mut col_idx: usize = 0;
    let mut start = 0;
    let mut row_base = spans.len(); // start of current row's slots

    // Pre-allocate a row's worth of sentinel spans
    spans.extend(std::iter::repeat(EMPTY_SPAN).take(stride));

    for (pos, &b) in input.iter().enumerate() {
        if b == b'\n' {
            if pos > start {
                // Cell boundary — record if this column is projected
                if col_idx <= max_col {
                    if let Some(&proj_idx) = col_map.get(col_idx) {
                        if proj_idx != usize::MAX {
                            spans[row_base + proj_idx] = CellSpan { start, end: pos };
                        }
                    }
                }
                col_idx += 1;
            } else {
                // Row boundary (\n\n) — finalise this row, start next
                col_idx = 0;
                row_base = spans.len();
                spans.extend(std::iter::repeat(EMPTY_SPAN).take(stride));
            }
            start = pos + 1;
        }
    }

    // Trailing cell without final LF
    if start < input.len() {
        if col_idx <= max_col {
            if let Some(&proj_idx) = col_map.get(col_idx) {
                if proj_idx != usize::MAX {
                    spans[row_base + proj_idx] = CellSpan {
                        start,
                        end: input.len(),
                    };
                }
            }
        }
        col_idx += 1;
    }

    // If we wrote any cells into the last row slot, count it
    let num_rows = if col_idx > 0 {
        spans.len() / stride
    } else {
        // Last row slot was pre-allocated but unused (input ended with \n\n)
        spans.truncate(spans.len() - stride);
        spans.len() / stride
    };

    (num_rows, spans)
}

/// Parallel projected indexing.
fn index_projected_parallel(
    input: &[u8],
    col_map: &[usize],
    max_col: usize,
    stride: usize,
) -> (usize, Vec<CellSpan>) {
    let num_threads = rayon::current_num_threads();
    let chunk_size = input.len() / num_threads;

    if chunk_size == 0 {
        return index_projected_sequential(input, col_map, max_col, stride);
    }

    let finder = memmem::Finder::new(b"\n\n");
    let mut splits = Vec::with_capacity(num_threads + 1);
    splits.push(0usize);

    for i in 1..num_threads {
        let nominal = i * chunk_size;
        if let Some(offset) = finder.find(&input[nominal..]) {
            let split = nominal + offset + 2;
            if split < input.len() {
                splits.push(split);
            }
        }
    }
    splits.push(input.len());
    splits.dedup();

    if splits.len() <= 2 {
        return index_projected_sequential(input, col_map, max_col, stride);
    }

    let chunks: Vec<&[u8]> = splits.windows(2).map(|w| &input[w[0]..w[1]]).collect();
    let offsets: Vec<usize> = splits[..splits.len() - 1].to_vec();

    let chunk_results: Vec<(usize, Vec<CellSpan>)> = chunks
        .par_iter()
        .zip(offsets.par_iter())
        .map(|(chunk, &base_offset)| {
            let (nrows, local_spans) =
                index_projected_sequential(chunk, col_map, max_col, stride);
            // Shift spans to absolute positions
            let shifted: Vec<CellSpan> = local_spans
                .into_iter()
                .map(|span| {
                    if span.start == span.end {
                        span // empty sentinel — leave as-is
                    } else {
                        CellSpan {
                            start: span.start + base_offset,
                            end: span.end + base_offset,
                        }
                    }
                })
                .collect();
            (nrows, shifted)
        })
        .collect();

    let total_rows: usize = chunk_results.iter().map(|(n, _)| n).sum();
    let mut result = Vec::with_capacity(total_rows * stride);
    for (_, chunk_spans) in chunk_results {
        result.extend(chunk_spans);
    }

    (total_rows, result)
}

// ── Single-pass projected decode (scan + unescape in one pass) ──────

/// Decode only the specified columns from raw bytes.
///
/// Single-pass: scans for cell/row boundaries and directly unescapes
/// only the cells in projected columns.  No intermediate structural index.
/// Each inner vec has exactly `columns.len()` entries (same order as `columns`).
pub fn decode_bytes_projected(input: &[u8], columns: &[usize]) -> Vec<Vec<Vec<u8>>> {
    if input.is_empty() || columns.is_empty() {
        return Vec::new();
    }

    if input.len() < PARALLEL_THRESHOLD {
        decode_projected_sequential(input, columns)
    } else {
        decode_projected_parallel(input, columns)
    }
}

/// Sequential single-pass projected decode.
fn decode_projected_sequential(input: &[u8], columns: &[usize]) -> Vec<Vec<Vec<u8>>> {
    let (col_map, max_col) = build_col_map(columns);
    let stride = columns.len();
    let mut data: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut row: Vec<Vec<u8>> = vec![Vec::new(); stride];
    let mut col_idx: usize = 0;
    let mut start = 0;
    let mut row_has_cells = false;

    for (pos, &b) in input.iter().enumerate() {
        if b == b'\n' {
            if pos > start {
                // Cell boundary
                if col_idx <= max_col {
                    if let Some(&proj_idx) = col_map.get(col_idx) {
                        if proj_idx != usize::MAX {
                            row[proj_idx] = unescape_bytes(&input[start..pos]);
                        }
                    }
                }
                col_idx += 1;
                row_has_cells = true;
            } else {
                // Row boundary
                if row_has_cells || !data.is_empty() || col_idx == 0 {
                    data.push(row);
                    row = vec![Vec::new(); stride];
                }
                col_idx = 0;
                row_has_cells = false;
            }
            start = pos + 1;
        }
    }

    // Trailing cell
    if start < input.len() {
        if col_idx <= max_col {
            if let Some(&proj_idx) = col_map.get(col_idx) {
                if proj_idx != usize::MAX {
                    row[proj_idx] = unescape_bytes(&input[start..]);
                }
            }
        }
        row_has_cells = true;
    }

    if row_has_cells {
        data.push(row);
    }

    data
}

/// Parallel single-pass projected decode.
fn decode_projected_parallel(input: &[u8], columns: &[usize]) -> Vec<Vec<Vec<u8>>> {
    let num_threads = rayon::current_num_threads();
    let chunk_size = input.len() / num_threads;

    if chunk_size == 0 {
        return decode_projected_sequential(input, columns);
    }

    let finder = memmem::Finder::new(b"\n\n");
    let mut splits = Vec::with_capacity(num_threads + 1);
    splits.push(0usize);

    for i in 1..num_threads {
        let nominal = i * chunk_size;
        if let Some(offset) = finder.find(&input[nominal..]) {
            let split = nominal + offset + 2;
            if split < input.len() {
                splits.push(split);
            }
        }
    }
    splits.push(input.len());
    splits.dedup();

    if splits.len() <= 2 {
        return decode_projected_sequential(input, columns);
    }

    let chunks: Vec<&[u8]> = splits.windows(2).map(|w| &input[w[0]..w[1]]).collect();

    let chunk_results: Vec<Vec<Vec<Vec<u8>>>> = chunks
        .par_iter()
        .map(|chunk| decode_projected_sequential(chunk, columns))
        .collect();

    let total_rows: usize = chunk_results.iter().map(|r| r.len()).sum();
    let mut result = Vec::with_capacity(total_rows);
    for chunk_rows in chunk_results {
        result.extend(chunk_rows);
    }
    result
}

/// Encode a seqseq into an NSV string.
pub fn encode(data: &[Vec<String>]) -> String {
    let mut result = Vec::new();

    for row in data {
        for cell in row {
            result.extend_from_slice(&escape_bytes(cell.as_bytes()));
            result.push(b'\n');
        }
        result.push(b'\n');
    }

    // Safety: encoding only inserts ASCII bytes (\, n, LF) — preserves UTF-8.
    String::from_utf8(result).unwrap()
}

/// Encode a seqseq of byte vectors into raw NSV bytes.
pub fn encode_bytes(data: &[Vec<Vec<u8>>]) -> Vec<u8> {
    let mut result = Vec::new();

    for row in data {
        for cell in row {
            result.extend_from_slice(&escape_bytes(cell));
            result.push(b'\n');
        }
        result.push(b'\n');
    }

    result
}

/// A single warning found during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    pub kind: WarningKind,
    pub pos: usize,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningKind {
    /// `\` followed by a byte other than `n` or `\`
    UnknownEscape(u8),
    /// `\` immediately before LF or at EOF
    DanglingBackslash,
    /// Non-empty input not ending with LF
    NoTerminalLf,
}

/// Report edge cases in raw NSV input without altering parsing behavior.
///
/// Warns on unknown escape sequences, dangling backslashes, and missing terminal LF.
/// Positions are byte offsets; line and col are 1-indexed.
pub fn check(input: &[u8]) -> Vec<Warning> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut warnings = Vec::new();
    let len = input.len();
    let mut line: usize = 1;
    let mut line_start: usize = 0;
    let mut escaped = false;

    for (i, &b) in input.iter().enumerate() {
        if escaped {
            match b {
                b'n' | b'\\' => {}
                b'\n' => warnings.push(Warning {
                    kind: WarningKind::DanglingBackslash,
                    pos: i - 1,
                    line,
                    col: i - line_start,
                }),
                _ => warnings.push(Warning {
                    kind: WarningKind::UnknownEscape(b),
                    pos: i - 1,
                    line,
                    col: i - line_start,
                }),
            }
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        }
        if b == b'\n' {
            line += 1;
            line_start = i + 1;
        }
    }

    if escaped {
        warnings.push(Warning {
            kind: WarningKind::DanglingBackslash,
            pos: len - 1,
            line,
            col: len - line_start,
        });
    }

    if line_start != len {
        warnings.push(Warning {
            kind: WarningKind::NoTerminalLf,
            pos: len,
            line,
            col: len - line_start + 1,
        });
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_table() {
        let nsv = "col1\ncol2\n\na\nb\n\nc\nd\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                vec!["col1".to_string(), "col2".to_string()],
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
            ]
        );
    }

    #[test]
    fn test_empty_fields() {
        let nsv = "a\n\\\nb\n\n\\\nc\n\\\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "".to_string(), "b".to_string()],
                vec!["".to_string(), "c".to_string(), "".to_string()],
            ]
        );
    }

    #[test]
    fn test_escape_sequences() {
        let nsv = "Line 1\\nLine 2\nBackslash: \\\\\nNot a newline: \\\\n\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![vec![
                "Line 1\nLine 2".to_string(),
                "Backslash: \\".to_string(),
                "Not a newline: \\n".to_string()
            ],]
        );
    }

    #[test]
    fn test_empty_rows() {
        let nsv = "first\n\n\n\nsecond\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                vec!["first".to_string()],
                vec![],
                vec![],
                vec!["second".to_string()],
            ]
        );
    }

    #[test]
    fn test_multiple_empty_rows() {
        let nsv = "a\n\n\n\n\nb\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                vec!["a".to_string()],
                vec![],
                vec![],
                vec![],
                vec!["b".to_string()],
            ]
        );
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![
            vec!["col1".to_string(), "col2".to_string()],
            vec!["a".to_string(), "b".to_string()],
            vec!["".to_string(), "value\\with\\backslash".to_string()],
            vec!["multi\nline".to_string(), "normal".to_string()],
        ];

        let encoded = encode(&original);
        let decoded = decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_unrecognized_escape() {
        let nsv = "\\x41\\t\\r\n";
        let result = decode(nsv);
        assert_eq!(result, vec![vec!["\\x41\\t\\r".to_string()],]);
    }

    #[test]
    fn test_dangling_backslash() {
        let nsv = "text\\\n";
        let result = decode(nsv);
        assert_eq!(result, vec![vec!["text".to_string()],]);
    }

    #[test]
    fn test_empty_input() {
        let result = decode("");
        assert_eq!(result, Vec::<Vec<String>>::new());
    }

    #[test]
    fn test_no_trailing_newline() {
        let nsv = "a\nb";
        let result = decode(nsv);
        assert_eq!(result, vec![vec!["a".to_string(), "b".to_string()],]);
    }

    #[test]
    fn test_only_empty_rows() {
        let nsv = "\n\n\n\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                Vec::<String>::new(),
                Vec::<String>::new(),
                Vec::<String>::new(),
                Vec::<String>::new(),
            ]
        );
    }

    #[test]
    fn test_starts_with_empty_row() {
        let nsv = "\n\nfirst\n";
        let result = decode(nsv);
        assert_eq!(
            result,
            vec![
                Vec::<String>::new(),
                Vec::<String>::new(),
                vec!["first".to_string()],
            ]
        );
    }

    #[test]
    fn test_large_file() {
        // Generate ~10MB of data to verify parallel path is exercised
        // (needs to exceed PARALLEL_THRESHOLD of 64KB)
        let large_data: Vec<Vec<String>> = (0..100_000)
            .map(|i| vec![format!("row{}", i), format!("data{}", i)])
            .collect();

        let encoded = encode(&large_data);

        // Verify it's large enough to trigger parallel parsing
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = decode(&encoded);
        assert_eq!(large_data, decoded);
    }

    #[test]
    fn test_parallel_with_empty_rows() {
        // Test parallel path with empty rows mixed in
        let mut data = Vec::new();

        // Create enough data to exceed 64KB threshold
        for i in 0..10_000 {
            data.push(vec![format!("value{}", i)]);

            // Add empty row every 100 rows
            if i % 100 == 0 {
                data.push(vec![]);
            }
        }

        let encoded = encode(&data);
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = decode(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_parallel_with_escape_sequences() {
        // Test parallel path with cells containing escape sequences
        let mut data = Vec::new();

        for i in 0..10_000 {
            data.push(vec![
                format!("Line 1\nLine 2 {}", i),
                format!("Backslash: \\ {}", i),
                "".to_string(),
            ]);
        }

        let encoded = encode(&data);
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = decode(&encoded);
        assert_eq!(data, decoded);
    }

    // ── Byte-level tests ──

    #[test]
    fn test_bytes_roundtrip() {
        let original: Vec<Vec<Vec<u8>>> = vec![
            vec![b"col1".to_vec(), b"col2".to_vec()],
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"".to_vec(), b"value\\with\\backslash".as_slice().to_vec()],
            vec![b"multi\nline".to_vec(), b"normal".to_vec()],
        ];

        let encoded = encode_bytes(&original);
        let decoded = decode_bytes(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_bytes_non_utf8() {
        // Latin-1 bytes with values > 0x7F — not valid UTF-8
        let cell1: Vec<u8> = vec![0xC0, 0xE9, 0xF1]; // àéñ in Latin-1
        let cell2: Vec<u8> = vec![0xFF, 0xFE, 0x80]; // arbitrary high bytes
        let original: Vec<Vec<Vec<u8>>> = vec![vec![cell1.clone(), cell2.clone()]];

        let encoded = encode_bytes(&original);
        let decoded = decode_bytes(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_bytes_empty_cells_and_rows() {
        let original: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec(), b"".to_vec(), b"b".to_vec()],
            vec![],
            vec![b"".to_vec()],
        ];

        let encoded = encode_bytes(&original);
        let decoded = decode_bytes(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_bytes_with_special_ascii_bytes() {
        // Verify that only 0x0A (LF), 0x5C (\), 0x6E (n) are structurally significant
        let cell: Vec<u8> = (0u8..=255)
            .filter(|&b| b != b'\n' && b != b'\\')
            .collect();
        let original = vec![vec![cell.clone()]];

        let encoded = encode_bytes(&original);
        let decoded = decode_bytes(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_decode_bytes_matches_decode_str() {
        // Confirm decode(s) and decode_bytes(s.as_bytes()) produce equivalent structures
        let inputs = vec![
            "col1\ncol2\n\na\nb\n\nc\nd\n",
            "a\n\\\nb\n\n\\\nc\n\\\n",
            "Line 1\\nLine 2\nBackslash: \\\\\n",
            "first\n\n\n\nsecond\n",
            "",
            "\n\n\n\n",
        ];

        for input in inputs {
            let str_result = decode(input);
            let byte_result = decode_bytes(input.as_bytes());

            // Convert byte result to strings for comparison
            let byte_as_str: Vec<Vec<String>> = byte_result
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| String::from_utf8(cell).unwrap())
                        .collect()
                })
                .collect();

            assert_eq!(str_result, byte_as_str, "mismatch for input: {:?}", input);
        }
    }

    #[test]
    fn test_bytes_large_parallel() {
        // Generate enough data to trigger parallel path
        let large_data: Vec<Vec<Vec<u8>>> = (0..100_000)
            .map(|i| {
                vec![
                    format!("row{}", i).into_bytes(),
                    format!("data{}", i).into_bytes(),
                ]
            })
            .collect();

        let encoded = encode_bytes(&large_data);
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = decode_bytes(&encoded);
        assert_eq!(large_data, decoded);
    }

    // ── check() tests ──

    #[test]
    fn test_check_empty_input() {
        assert_eq!(check(b""), vec![]);
    }

    #[test]
    fn test_check_just_lf() {
        assert_eq!(check(b"\n"), vec![]);
    }

    #[test]
    fn test_check_no_issues() {
        assert_eq!(check(b"col1\ncol2\n\na\nb\n\n"), vec![]);
        assert_eq!(check(b"hello\\\\world\n\\n\n\n"), vec![]);
    }

    #[test]
    fn test_check_single_unknown_escape() {
        let warnings = check(b"hello\\tworld\n");
        assert_eq!(
            warnings,
            vec![Warning {
                kind: WarningKind::UnknownEscape(b't'),
                pos: 5,
                line: 1,
                col: 6,
            }]
        );
    }

    #[test]
    fn test_check_multiple_unknown_escapes_different_lines() {
        let warnings = check(b"\\thello\n\\rworld\n\n");
        assert_eq!(
            warnings,
            vec![
                Warning {
                    kind: WarningKind::UnknownEscape(b't'),
                    pos: 0,
                    line: 1,
                    col: 1,
                },
                Warning {
                    kind: WarningKind::UnknownEscape(b'r'),
                    pos: 8,
                    line: 2,
                    col: 1,
                },
            ]
        );
    }

    #[test]
    fn test_check_dangling_backslash_mid_file() {
        let warnings = check(b"text\\\nmore\n\n");
        assert_eq!(
            warnings,
            vec![Warning {
                kind: WarningKind::DanglingBackslash,
                pos: 4,
                line: 1,
                col: 5,
            }]
        );
    }

    #[test]
    fn test_check_dangling_backslash_at_eof() {
        let warnings = check(b"text\\");
        assert_eq!(
            warnings,
            vec![
                Warning {
                    kind: WarningKind::DanglingBackslash,
                    pos: 4,
                    line: 1,
                    col: 5,
                },
                Warning {
                    kind: WarningKind::NoTerminalLf,
                    pos: 5,
                    line: 1,
                    col: 6,
                },
            ]
        );
    }

    #[test]
    fn test_check_no_terminal_lf() {
        let warnings = check(b"hello");
        assert_eq!(
            warnings,
            vec![Warning {
                kind: WarningKind::NoTerminalLf,
                pos: 5,
                line: 1,
                col: 6,
            }]
        );
    }

    #[test]
    fn test_check_combination() {
        // \(0) t(1) h(2) e(3) l(4) l(5) o(6) \(7) LF(8) w(9) o(10) r(11) l(12) d(13)
        let warnings = check(b"\\thello\\\nworld");
        assert_eq!(
            warnings,
            vec![
                Warning {
                    kind: WarningKind::UnknownEscape(b't'),
                    pos: 0,
                    line: 1,
                    col: 1,
                },
                Warning {
                    kind: WarningKind::DanglingBackslash,
                    pos: 7,
                    line: 1,
                    col: 8,
                },
                Warning {
                    kind: WarningKind::NoTerminalLf,
                    pos: 14,
                    line: 2,
                    col: 6,
                },
            ]
        );
    }

    #[test]
    fn test_check_non_utf8() {
        // Non-UTF-8 bytes with a bad escape: 0xFF 0xFE \t 0x80 LF
        let input: &[u8] = &[0xFF, 0xFE, b'\\', b't', 0x80, b'\n'];
        let warnings = check(input);
        assert_eq!(
            warnings,
            vec![Warning {
                kind: WarningKind::UnknownEscape(b't'),
                pos: 2,
                line: 1,
                col: 3,
            }]
        );
    }

    // ── LazyNsv / projected tests ──

    #[test]
    fn test_lazy_matches_decode() {
        let nsv = b"col1\ncol2\ncol3\n\na\nb\nc\n\nd\ne\nf\n\n";
        let lazy = decode_lazy(nsv);
        let full = decode_bytes(nsv);

        assert_eq!(lazy.row_count(), full.len());
        for (ri, row) in full.iter().enumerate() {
            assert_eq!(lazy.col_count(ri), row.len());
            for (ci, cell) in row.iter().enumerate() {
                assert_eq!(lazy.get(ri, ci).unwrap(), *cell);
            }
        }
    }

    #[test]
    fn test_lazy_empty_input() {
        let lazy = decode_lazy(b"");
        assert_eq!(lazy.row_count(), 0);
        assert!(lazy.get(0, 0).is_none());
    }

    #[test]
    fn test_lazy_empty_cells() {
        let nsv = b"a\n\\\nb\n\n\\\nc\n\\\n\n";
        let lazy = decode_lazy(nsv);
        let full = decode_bytes(nsv);

        assert_eq!(lazy.row_count(), full.len());
        for (ri, row) in full.iter().enumerate() {
            for (ci, cell) in row.iter().enumerate() {
                assert_eq!(lazy.get(ri, ci).unwrap(), *cell);
            }
        }
    }

    #[test]
    fn test_lazy_escape_sequences() {
        let nsv = b"Line 1\\nLine 2\nBackslash: \\\\\n\n";
        let lazy = decode_lazy(nsv);
        assert_eq!(lazy.get(0, 0).unwrap(), b"Line 1\nLine 2");
        assert_eq!(lazy.get(0, 1).unwrap(), b"Backslash: \\");
    }

    #[test]
    fn test_lazy_get_raw() {
        let nsv = b"Line 1\\nLine 2\nhello\n\n";
        let lazy = decode_lazy(nsv);
        // Raw should be the escaped form
        assert_eq!(lazy.get_raw(0, 0).unwrap(), b"Line 1\\nLine 2");
        // No escaping needed
        assert_eq!(lazy.get_raw(0, 1).unwrap(), b"hello");
    }

    #[test]
    fn test_project_subset() {
        let nsv = b"c0\nc1\nc2\nc3\n\na\nb\nc\nd\n\ne\nf\ng\nh\n\n";
        let lazy = decode_lazy(nsv);

        // Project only columns 0 and 2
        let projected = lazy.project(&[0, 2]);
        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0], vec![b"c0".to_vec(), b"c2".to_vec()]);
        assert_eq!(projected[1], vec![b"a".to_vec(), b"c".to_vec()]);
        assert_eq!(projected[2], vec![b"e".to_vec(), b"g".to_vec()]);
    }

    #[test]
    fn test_project_single_column() {
        let nsv = b"name\nage\nsalary\n\nAlice\n30\n50000\n\nBob\n25\n75000\n\n";
        let lazy = decode_lazy(nsv);

        // Project only column 1 (age)
        let projected = lazy.project(&[1]);
        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0], vec![b"age".to_vec()]);
        assert_eq!(projected[1], vec![b"30".to_vec()]);
        assert_eq!(projected[2], vec![b"25".to_vec()]);
    }

    #[test]
    fn test_project_reorder() {
        let nsv = b"a\nb\nc\n\n1\n2\n3\n\n";
        let lazy = decode_lazy(nsv);

        // Columns in reversed order
        let projected = lazy.project(&[2, 0]);
        assert_eq!(projected[0], vec![b"c".to_vec(), b"a".to_vec()]);
        assert_eq!(projected[1], vec![b"3".to_vec(), b"1".to_vec()]);
    }

    #[test]
    fn test_project_out_of_range() {
        let nsv = b"a\nb\n\n";
        let lazy = decode_lazy(nsv);

        // Column 5 is out of range — should produce empty vec
        let projected = lazy.project(&[0, 5]);
        assert_eq!(projected[0], vec![b"a".to_vec(), b"".to_vec()]);
    }

    #[test]
    fn test_decode_bytes_projected_convenience() {
        let nsv = b"c0\nc1\nc2\n\na\nb\nc\n\n";
        let projected = decode_bytes_projected(nsv, &[1]);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0], vec![b"c1".to_vec()]);
        assert_eq!(projected[1], vec![b"b".to_vec()]);
    }

    #[test]
    fn test_lazy_large_parallel() {
        // Generate enough data to trigger parallel indexing
        let large_data: Vec<Vec<String>> = (0..100_000)
            .map(|i| vec![format!("row{}", i), format!("data{}", i)])
            .collect();
        let encoded = encode(&large_data);
        let encoded_bytes = encoded.as_bytes();
        assert!(encoded_bytes.len() > PARALLEL_THRESHOLD);

        let lazy = decode_lazy(encoded_bytes);
        let full = decode_bytes(encoded_bytes);

        assert_eq!(lazy.row_count(), full.len());
        // Spot-check some rows
        for ri in [0, 1, 1000, 50_000, 99_999] {
            assert_eq!(lazy.col_count(ri), full[ri].len());
            for ci in 0..full[ri].len() {
                assert_eq!(lazy.get(ri, ci).unwrap(), full[ri][ci]);
            }
        }
    }

    #[test]
    fn test_lazy_parallel_with_empty_rows() {
        let mut data = Vec::new();
        for i in 0..10_000 {
            data.push(vec![format!("value{}", i)]);
            if i % 100 == 0 {
                data.push(vec![]);
            }
        }
        let encoded = encode(&data);
        let encoded_bytes = encoded.as_bytes();
        assert!(encoded_bytes.len() > PARALLEL_THRESHOLD);

        let lazy = decode_lazy(encoded_bytes);
        let full = decode_bytes(encoded_bytes);

        assert_eq!(lazy.row_count(), full.len());
        for (ri, row) in full.iter().enumerate() {
            assert_eq!(lazy.col_count(ri), row.len());
            for (ci, cell) in row.iter().enumerate() {
                assert_eq!(lazy.get(ri, ci).unwrap(), *cell);
            }
        }
    }

    #[test]
    fn test_project_with_escapes_parallel() {
        let mut data = Vec::new();
        for i in 0..10_000 {
            data.push(vec![
                format!("Line 1\nLine 2 {}", i),
                format!("Backslash: \\ {}", i),
                format!("plain{}", i),
            ]);
        }
        let encoded = encode(&data);
        let encoded_bytes = encoded.as_bytes();
        assert!(encoded_bytes.len() > PARALLEL_THRESHOLD);

        // Project only the last column (no escaping needed)
        let projected = decode_bytes_projected(encoded_bytes, &[2]);
        assert_eq!(projected.len(), data.len());
        for (ri, row) in data.iter().enumerate() {
            assert_eq!(
                String::from_utf8(projected[ri][0].clone()).unwrap(),
                row[2]
            );
        }

        // Project all columns and verify matches full decode
        let full = decode_bytes(encoded_bytes);
        let projected_all = decode_bytes_projected(encoded_bytes, &[0, 1, 2]);
        assert_eq!(projected_all, full);
    }
}
