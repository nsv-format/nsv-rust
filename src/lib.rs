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

/// Threshold for using parallel parsing (64KB)
const PARALLEL_THRESHOLD: usize = 64 * 1024;

pub fn loads(s: &str) -> Vec<Vec<String>> {
    if s.is_empty() {
        return Vec::new();
    }

    // For small inputs, use sequential fast path
    if s.len() < PARALLEL_THRESHOLD {
        return loads_sequential(s);
    }

    loads_parallel(s)
}

/// Sequential implementation for small files
fn loads_sequential(s: &str) -> Vec<Vec<String>> {
    let mut data = Vec::new();
    let mut row = Vec::new();
    let mut start = 0;

    let chars: Vec<char> = s.chars().collect();

    for (pos, &c) in chars.iter().enumerate() {
        if c == '\n' {
            if pos > start {
                let cell_text: String = chars[start..pos].iter().collect();
                row.push(unescape(&cell_text));
            } else {
                data.push(row);
                row = Vec::new();
            }
            start = pos + 1;
        }
    }

    if start < chars.len() {
        let cell_text: String = chars[start..].iter().collect();
        row.push(unescape(&cell_text));
    }

    if !row.is_empty() {
        data.push(row);
    }

    data
}

/// Parallel implementation for large files
fn loads_parallel(s: &str) -> Vec<Vec<String>> {
    // Use fast memchr SIMD scan to find all \n\n boundaries
    // Track which boundaries indicate empty rows during the scan
    let bytes = s.as_bytes();
    let finder = memmem::Finder::new(b"\n\n");
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while let Some(offset) = finder.find(&bytes[pos..]) {
        let abs_pos = pos + offset;
        // abs_pos points to the first \n of \n\n
        // The row ends after both newlines, so boundary is at abs_pos
        boundaries.push(abs_pos);

        // Check for consecutive empty rows: when we have \n\n\n...
        // Each additional \n after the initial \n\n represents another empty row
        let mut check_pos = abs_pos + 2;
        while check_pos < bytes.len() && bytes[check_pos] == b'\n' {
            // Found another \n, meaning another \n\n pattern (overlapping)
            boundaries.push(check_pos - 1);
            check_pos += 1;
        }

        // Continue searching after all the consecutive newlines we just processed
        pos = check_pos;
    }

    // Edge case: if no double newlines found, treat as single row
    if boundaries.is_empty() {
        let row = parse_row(s);
        return if row.is_empty() {
            Vec::new()
        } else {
            vec![row]
        };
    }

    // Build row slices, handling potentially overlapping boundaries from empty rows
    let mut row_slices = Vec::new();
    let mut start = 0;

    for &boundary in &boundaries {
        // For overlapping boundaries (from empty rows), boundary might be < start
        // In that case, we have an empty row
        if boundary < start {
            row_slices.push("");
            // Still need to advance start past this empty row's \n\n
            start = boundary + 2;
        } else {
            row_slices.push(&s[start..boundary]);
            start = boundary + 2;
        }
    }

    // Handle remaining data after final "\n\n" boundary (if any)
    if start < s.len() {
        row_slices.push(&s[start..]);
    }

    // Parse rows in parallel - fast path, no string contains checks
    row_slices
        .par_iter()
        .map(|&slice| parse_row(slice))
        .collect()
}

/// Parse a single row from a string slice
fn parse_row(row_str: &str) -> Vec<String> {
    if row_str.is_empty() {
        return Vec::new();
    }

    let mut cells = Vec::new();
    let mut start = 0;

    for (pos, c) in row_str.char_indices() {
        if c == '\n' {
            if pos > start {
                let cell_text = &row_str[start..pos];
                cells.push(unescape(cell_text));
            } else {
                // Empty cell at position (consecutive newlines within a row shouldn't happen in valid NSV)
                cells.push(String::new());
            }
            start = pos + 1;
        }
    }

    // Handle last cell if no trailing newline
    if start < row_str.len() {
        let cell_text = &row_str[start..];
        cells.push(unescape(cell_text));
    }

    cells
}

/// Unescape a single NSV cell value.
///
/// Interprets `\` as empty string, `\\` as literal backslash, and `\n` as newline.
/// Unrecognized escape sequences are passed through with the literal backslash.
/// Dangling backslash at end of string is stripped.
pub fn unescape(s: &str) -> String {
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

/// Escape a single NSV cell value.
///
/// Empty strings become `\`, backslashes become `\\`, newlines become `\n`.
/// Strings without special characters are returned as-is.
pub fn escape(s: &str) -> String {
    if s.is_empty() {
        return "\\".to_string();
    }

    if s.contains('\n') || s.contains('\\') {
        s.replace('\\', "\\\\").replace('\n', "\\n")
    } else {
        s.to_string()
    }
}

pub fn dumps(data: &[Vec<String>]) -> String {
    let mut result = String::new();

    for row in data {
        for cell in row {
            result.push_str(&escape(cell));
            result.push('\n');
        }
        result.push('\n');
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_table() {
        let nsv = "col1\ncol2\n\na\nb\n\nc\nd\n";
        let result = loads(nsv);
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
        let result = loads(nsv);
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
        let result = loads(nsv);
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
        let result = loads(nsv);
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
        let result = loads(nsv);
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

        let encoded = dumps(&original);
        let decoded = loads(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_unrecognized_escape() {
        let nsv = "\\x41\\t\\r\n";
        let result = loads(nsv);
        assert_eq!(result, vec![vec!["\\x41\\t\\r".to_string()],]);
    }

    #[test]
    fn test_dangling_backslash() {
        let nsv = "text\\\n";
        let result = loads(nsv);
        assert_eq!(result, vec![vec!["text".to_string()],]);
    }

    #[test]
    fn test_empty_input() {
        let result = loads("");
        assert_eq!(result, Vec::<Vec<String>>::new());
    }

    #[test]
    fn test_no_trailing_newline() {
        let nsv = "a\nb";
        let result = loads(nsv);
        assert_eq!(result, vec![vec!["a".to_string(), "b".to_string()],]);
    }

    #[test]
    fn test_only_empty_rows() {
        let nsv = "\n\n\n\n";
        let result = loads(nsv);
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
        let result = loads(nsv);
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

        let encoded = dumps(&large_data);

        // Verify it's large enough to trigger parallel parsing
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = loads(&encoded);
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

        let encoded = dumps(&data);
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = loads(&encoded);
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

        let encoded = dumps(&data);
        assert!(encoded.len() > PARALLEL_THRESHOLD);

        let decoded = loads(&encoded);
        assert_eq!(data, decoded);
    }
}
