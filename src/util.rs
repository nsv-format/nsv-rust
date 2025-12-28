//! NSV utility functions for structural operations and escaping.
//!
//! These functions provide low-level building blocks for NSV encoding/decoding:
//! - `escape_seqseq` / `unescape_seqseq`: Apply escaping at depth 2
//! - `spill` / `unspill`: Structural dimension operations
//!
//! The encoding pipeline is: `encode = spill('\n') ∘ spill("") ∘ escape_seqseq`
//! The decoding pipeline is: `decode = unescape_seqseq ∘ unspill("") ∘ unspill('\n')`

use crate::{escape, unescape};

/// Apply NSV escaping at depth 2: map(map(escape)).
///
/// Takes a 2D structure of strings and escapes each cell.
///
/// # Example
/// ```
/// use nsv::util::escape_seqseq;
///
/// let data = vec![
///     vec!["hello".to_string(), "world\nnewline".to_string()],
/// ];
/// let escaped = escape_seqseq(&data);
/// assert_eq!(escaped[0][0], "hello");
/// assert_eq!(escaped[0][1], "world\\nnewline");
/// ```
pub fn escape_seqseq(seqseq: &[Vec<String>]) -> Vec<Vec<String>> {
    seqseq
        .iter()
        .map(|row| row.iter().map(|cell| escape(cell)).collect())
        .collect()
}

/// Apply NSV unescaping at depth 2: map(map(unescape)).
///
/// Takes a 2D structure of escaped strings and unescapes each cell.
///
/// # Example
/// ```
/// use nsv::util::unescape_seqseq;
///
/// let data = vec![
///     vec!["hello".to_string(), "world\\nnewline".to_string()],
/// ];
/// let unescaped = unescape_seqseq(&data);
/// assert_eq!(unescaped[0][0], "hello");
/// assert_eq!(unescaped[0][1], "world\nnewline");
/// ```
pub fn unescape_seqseq(seqseq: &[Vec<String>]) -> Vec<Vec<String>> {
    seqseq
        .iter()
        .map(|row| row.iter().map(|cell| unescape(cell)).collect())
        .collect()
}

/// Collapse a dimension of seqseq by spilling termination markers into the resulting flat sequence.
///
/// Pure structural operation - does NOT perform escaping.
///
/// In the NSV encoding pipeline:
/// `encode = spill('\n') ∘ spill("") ∘ escape_seqseq`
///
/// # Example
/// ```
/// use nsv::util::spill;
///
/// let data = vec![
///     vec!["a".to_string(), "b".to_string()],
///     vec!["c".to_string()],
/// ];
/// let flat = spill(&data, String::new());
/// assert_eq!(flat, vec!["a", "b", "", "c", ""]);
/// ```
pub fn spill<T: Clone>(seqseq: &[Vec<T>], marker: T) -> Vec<T> {
    let mut seq = Vec::new();
    for row in seqseq {
        for item in row {
            seq.push(item.clone());
        }
        seq.push(marker.clone());
    }
    seq
}

/// Recover a dimension by picking up termination markers from the provided sequence.
///
/// Pure structural operation - does NOT perform unescaping.
///
/// In the NSV decoding pipeline:
/// `decode = unescape_seqseq ∘ unspill("") ∘ unspill('\n')`
///
/// Note: Strict mode - incomplete rows (without trailing marker) are discarded.
///
/// # Example
/// ```
/// use nsv::util::unspill;
///
/// let flat = vec!["a".to_string(), "b".to_string(), "".to_string(), "c".to_string(), "".to_string()];
/// let data = unspill(&flat, &String::new());
/// assert_eq!(data, vec![
///     vec!["a".to_string(), "b".to_string()],
///     vec!["c".to_string()],
/// ]);
/// ```
pub fn unspill<T: Clone + PartialEq>(seq: &[T], marker: &T) -> Vec<Vec<T>> {
    let mut seqseq = Vec::new();
    let mut row = Vec::new();
    for item in seq {
        if item != marker {
            row.push(item.clone());
        } else {
            seqseq.push(row);
            row = Vec::new();
        }
    }
    // Strict: don't append incomplete rows
    seqseq
}

/// Convenience function to spill characters with newline marker.
///
/// This is the second stage of NSV encoding: converting strings to a flat character sequence.
///
/// # Example
/// ```
/// use nsv::util::spill_chars;
///
/// let strings = vec!["ab".to_string(), "c".to_string(), "".to_string()];
/// let chars: String = spill_chars(&strings).into_iter().collect();
/// assert_eq!(chars, "ab\nc\n\n");
/// ```
pub fn spill_chars(strings: &[String]) -> Vec<char> {
    let mut chars = Vec::new();
    for s in strings {
        for c in s.chars() {
            chars.push(c);
        }
        chars.push('\n');
    }
    chars
}

/// Convenience function to unspill characters with newline marker.
///
/// This is the first stage of NSV decoding: splitting a character sequence into strings.
///
/// # Example
/// ```
/// use nsv::util::unspill_chars;
///
/// let chars: Vec<char> = "ab\nc\n\n".chars().collect();
/// let strings = unspill_chars(&chars);
/// assert_eq!(strings, vec!["ab".to_string(), "c".to_string(), "".to_string()]);
/// ```
pub fn unspill_chars(chars: &[char]) -> Vec<String> {
    let mut strings = Vec::new();
    let mut current = String::new();
    for &c in chars {
        if c != '\n' {
            current.push(c);
        } else {
            strings.push(current);
            current = String::new();
        }
    }
    // Strict: don't append incomplete strings
    strings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_seqseq() {
        let input = vec![
            vec!["hello".to_string(), "world\nnewline".to_string()],
            vec!["back\\slash".to_string(), "".to_string()],
        ];
        let result = escape_seqseq(&input);
        assert_eq!(
            result,
            vec![
                vec!["hello".to_string(), "world\\nnewline".to_string()],
                vec!["back\\\\slash".to_string(), "\\".to_string()],
            ]
        );
    }

    #[test]
    fn test_unescape_seqseq() {
        let input = vec![
            vec!["hello".to_string(), "world\\nnewline".to_string()],
            vec!["back\\\\slash".to_string(), "\\".to_string()],
        ];
        let result = unescape_seqseq(&input);
        assert_eq!(
            result,
            vec![
                vec!["hello".to_string(), "world\nnewline".to_string()],
                vec!["back\\slash".to_string(), "".to_string()],
            ]
        );
    }

    #[test]
    fn test_escape_unescape_roundtrip() {
        let original = vec![
            vec![
                "normal".to_string(),
                "with\nnewline".to_string(),
                "with\\backslash".to_string(),
            ],
            vec!["".to_string(), "both\n\\".to_string()],
        ];
        let escaped = escape_seqseq(&original);
        let unescaped = unescape_seqseq(&escaped);
        assert_eq!(original, unescaped);
    }

    #[test]
    fn test_spill_strings() {
        let input = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];
        let result = spill(&input, String::new());
        assert_eq!(
            result,
            vec![
                "a".to_string(),
                "b".to_string(),
                "".to_string(),
                "c".to_string(),
                "".to_string()
            ]
        );
    }

    #[test]
    fn test_spill_empty_rows() {
        let input: Vec<Vec<String>> = vec![vec![], vec!["a".to_string()], vec![]];
        let result = spill(&input, String::new());
        assert_eq!(
            result,
            vec![
                "".to_string(),
                "a".to_string(),
                "".to_string(),
                "".to_string()
            ]
        );
    }

    #[test]
    fn test_unspill_strings() {
        let input = vec![
            "a".to_string(),
            "b".to_string(),
            "".to_string(),
            "c".to_string(),
            "".to_string(),
        ];
        let result = unspill(&input, &String::new());
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string()],
            ]
        );
    }

    #[test]
    fn test_unspill_with_empty_rows() {
        let input = vec![
            "".to_string(),
            "a".to_string(),
            "".to_string(),
            "".to_string(),
        ];
        let result = unspill(&input, &String::new());
        assert_eq!(
            result,
            vec![
                Vec::<String>::new(),
                vec!["a".to_string()],
                Vec::<String>::new(),
            ]
        );
    }

    #[test]
    fn test_unspill_discards_incomplete() {
        let input = vec!["a".to_string(), "b".to_string()]; // No trailing marker
        let result = unspill(&input, &String::new());
        assert_eq!(result, Vec::<Vec<String>>::new()); // Empty - incomplete row discarded
    }

    #[test]
    fn test_spill_unspill_roundtrip() {
        let original = vec![
            vec!["a".to_string(), "b".to_string()],
            vec![],
            vec!["c".to_string(), "d".to_string(), "e".to_string()],
        ];
        let spilled = spill(&original, String::new());
        let unspilled = unspill(&spilled, &String::new());
        assert_eq!(original, unspilled);
    }

    #[test]
    fn test_spill_chars() {
        let input = vec!["ab".to_string(), "c".to_string(), "".to_string()];
        let result = spill_chars(&input);
        let as_string: String = result.into_iter().collect();
        assert_eq!(as_string, "ab\nc\n\n");
    }

    #[test]
    fn test_unspill_chars() {
        let input: Vec<char> = "ab\nc\n\n".chars().collect();
        let result = unspill_chars(&input);
        assert_eq!(
            result,
            vec!["ab".to_string(), "c".to_string(), "".to_string()]
        );
    }

    #[test]
    fn test_chars_roundtrip() {
        let original = vec![
            "hello".to_string(),
            "world".to_string(),
            "".to_string(),
            "test".to_string(),
        ];
        let spilled = spill_chars(&original);
        let unspilled = unspill_chars(&spilled);
        assert_eq!(original, unspilled);
    }

    #[test]
    fn test_full_encode_decode_pipeline() {
        // Test the complete pipeline:
        // encode = spill_chars ∘ spill("") ∘ escape_seqseq
        // decode = unescape_seqseq ∘ unspill("") ∘ unspill_chars

        let original = vec![
            vec!["hello".to_string(), "world\nnewline".to_string()],
            vec!["back\\slash".to_string()],
        ];

        // Encode pipeline
        let escaped = escape_seqseq(&original);
        let spilled_strings = spill(&escaped, String::new());
        let spilled_chars = spill_chars(&spilled_strings);
        let encoded: String = spilled_chars.into_iter().collect();

        // Decode pipeline
        let chars: Vec<char> = encoded.chars().collect();
        let unspilled_strings = unspill_chars(&chars);
        let unspilled_rows = unspill(&unspilled_strings, &String::new());
        let decoded = unescape_seqseq(&unspilled_rows);

        assert_eq!(original, decoded);
    }
}
