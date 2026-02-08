//! NSV (Newline-Separated Values) format implementation for Rust
//!
//! Fast parallel implementation using rayon and memchr.
//! See https://nsv-format.org for the specification.
//!
//! ## Parallel Parsing Strategy
//!
//! For files larger than 64KB, we use a parallel approach:
//! 1. Find all row boundaries (positions of `\n\n`) using memchr's SIMD-optimized search
//! 2. Split the input into row slices
//! 3. Parse each row in parallel using rayon
//! 4. Each row is split on `\n` and cells are unescaped
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

/// Parallel implementation for large inputs (byte-level).
fn decode_bytes_parallel(input: &[u8]) -> Vec<Vec<Vec<u8>>> {
    // Use fast memchr SIMD scan to find all \n\n boundaries
    // Track which boundaries indicate empty rows during the scan
    let finder = memmem::Finder::new(b"\n\n");
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while let Some(offset) = finder.find(&input[pos..]) {
        let abs_pos = pos + offset;
        // abs_pos points to the first \n of \n\n
        // The row ends after both newlines, so boundary is at abs_pos
        boundaries.push(abs_pos);

        // Check for consecutive empty rows: when we have \n\n\n...
        // Each additional \n after the initial \n\n represents another empty row
        let mut check_pos = abs_pos + 2;
        while check_pos < input.len() && input[check_pos] == b'\n' {
            // Found another \n, meaning another \n\n pattern (overlapping)
            boundaries.push(check_pos - 1);
            check_pos += 1;
        }

        // Continue searching after all the consecutive newlines we just processed
        pos = check_pos;
    }

    // Edge case: if no double newlines found, treat as single row
    if boundaries.is_empty() {
        let row = parse_row_bytes(input);
        return if row.is_empty() {
            Vec::new()
        } else {
            vec![row]
        };
    }

    // Build row slices, handling potentially overlapping boundaries from empty rows
    let mut row_slices: Vec<&[u8]> = Vec::new();
    let mut start = 0;

    for &boundary in &boundaries {
        // For overlapping boundaries (from empty rows), boundary might be < start
        // In that case, we have an empty row
        if boundary < start {
            row_slices.push(b"");
            // Still need to advance start past this empty row's \n\n
            start = boundary + 2;
        } else {
            row_slices.push(&input[start..boundary]);
            start = boundary + 2;
        }
    }

    // Handle remaining data after final "\n\n" boundary (if any)
    if start < input.len() {
        row_slices.push(&input[start..]);
    }

    // Parse rows in parallel - fast path, no string contains checks
    row_slices
        .par_iter()
        .map(|&slice| parse_row_bytes(slice))
        .collect()
}

/// Parse a single row from a byte slice.
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
                // Empty cell at position (consecutive newlines within a row shouldn't happen in valid NSV)
                cells.push(Vec::new());
            }
            start = pos + 1;
        }
    }

    // Handle last cell if no trailing newline
    if start < row.len() {
        cells.push(unescape_bytes(&row[start..]));
    }

    cells
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
}
