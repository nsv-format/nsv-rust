//! Utility functions for NSV structural operations.
//!
//! Encoding: `spill('\n') ∘ spill("") ∘ escape_seqseq`
//! Decoding: `unescape_seqseq ∘ unspill("") ∘ unspill('\n')`

use crate::{escape, unescape};

/// Apply NSV escaping at depth 2: map(map(escape)).
pub fn escape_seqseq(seqseq: &[Vec<String>]) -> Vec<Vec<String>> {
    seqseq
        .iter()
        .map(|row| row.iter().map(|cell| escape(cell)).collect())
        .collect()
}

/// Apply NSV unescaping at depth 2: map(map(unescape)).
pub fn unescape_seqseq(seqseq: &[Vec<String>]) -> Vec<Vec<String>> {
    seqseq
        .iter()
        .map(|row| row.iter().map(|cell| unescape(cell)).collect())
        .collect()
}

/// Collapse a dimension by spilling termination markers into the flat sequence.
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

/// Recover a dimension by picking up termination markers. Incomplete rows discarded.
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
    seqseq
}

/// Spill strings to chars with newline marker.
pub fn spill_chars(strings: &[String]) -> Vec<char> {
    let mut chars = Vec::new();
    for s in strings {
        chars.extend(s.chars());
        chars.push('\n');
    }
    chars
}

/// Unspill chars to strings with newline marker.
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
    strings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_unescape_seqseq() {
        let original = vec![
            vec!["normal".into(), "with\nnewline".into(), "with\\backslash".into()],
            vec!["".into(), "both\n\\".into()],
        ];
        let escaped = escape_seqseq(&original);
        assert_eq!(escaped[0][1], "with\\nnewline");
        assert_eq!(escaped[1][0], "\\");
        assert_eq!(unescape_seqseq(&escaped), original);
    }

    #[test]
    fn test_spill_unspill_invertibility() {
        let cases: Vec<Vec<Vec<String>>> = vec![
            vec![],
            vec![vec![]],
            vec![vec![], vec![]],
            vec![vec!["a".into()]],
            vec![vec!["a".into(), "b".into()], vec!["c".into()]],
            vec![vec!["a".into()], vec![], vec!["b".into()]],
        ];
        for seqseq in cases {
            let spilled = spill(&seqseq, String::new());
            assert_eq!(unspill(&spilled, &String::new()), seqseq);
        }

        // Also test with different types
        let ints: Vec<Vec<i32>> = vec![vec![1, 2], vec![3]];
        assert_eq!(spill(&ints, -1), vec![1, 2, -1, 3, -1]);
    }

    #[test]
    fn test_unspill_discards_incomplete() {
        let input = vec!["a".into(), "b".into()];
        assert!(unspill(&input, &String::new()).is_empty());
    }

    fn get_samples() -> Vec<(&'static str, Vec<Vec<String>>)> {
        vec![
            ("empty", vec![]),
            ("empty_one", vec![vec![]]),
            ("empty_rows", vec![vec![], vec![], vec![]]),
            ("basic", vec![vec!["a".into(), "b".into()], vec!["c".into(), "d".into()]]),
            ("empty_fields", vec![vec!["".into(), "x".into(), "".into()]]),
            ("special_chars", vec![
                vec!["field\\with\\backslashes".into(), "field\nwith\nnewlines".into()],
            ]),
            ("escape_edge_cases", vec![
                vec!["\\n".into(), "\\\n".into(), "\\\\n".into()],
                vec!["\\\\".into(), "\n\n".into()],
            ]),
        ]
    }

    #[test]
    fn test_decomposition_matches_direct() {
        for (name, seqseq) in get_samples() {
            // Encode via decomposition
            let escaped = escape_seqseq(&seqseq);
            let spilled = spill(&escaped, String::new());
            let encoded: String = spill_chars(&spilled).into_iter().collect();
            assert_eq!(crate::dumps(&seqseq), encoded, "encode mismatch: {}", name);

            // Decode via decomposition
            let chars: Vec<char> = encoded.chars().collect();
            let unspilled = unspill(&unspill_chars(&chars), &String::new());
            let decoded = unescape_seqseq(&unspilled);
            assert_eq!(seqseq, decoded, "roundtrip mismatch: {}", name);
        }
    }

    const PARALLEL_THRESHOLD: usize = 64 * 1024;

    fn verify_large_roundtrip(original: &[Vec<String>]) {
        let direct_encoded = crate::dumps(original);
        assert!(direct_encoded.len() > PARALLEL_THRESHOLD);

        let escaped = escape_seqseq(original);
        let spilled = spill(&escaped, String::new());
        let decomposed: String = spill_chars(&spilled).into_iter().collect();
        assert_eq!(direct_encoded, decomposed);

        let chars: Vec<char> = decomposed.chars().collect();
        let decoded = unescape_seqseq(&unspill(&unspill_chars(&chars), &String::new()));
        assert_eq!(original, decoded);
        assert_eq!(crate::loads(&direct_encoded), decoded);
    }

    #[test]
    fn test_large_datasets() {
        // Simple large dataset
        let simple: Vec<Vec<String>> = (0..10_000)
            .map(|i| vec![format!("row{}", i), format!("data{}", i)])
            .collect();
        verify_large_roundtrip(&simple);

        // With escapes
        let with_escapes: Vec<Vec<String>> = (0..10_000)
            .map(|i| vec![format!("line\n{}", i), format!("back\\{}", i), "".into()])
            .collect();
        verify_large_roundtrip(&with_escapes);

        // With empty rows
        let mut with_empty = Vec::new();
        for i in 0..10_000 {
            with_empty.push(vec![format!("v{}", i)]);
            if i % 100 == 0 {
                with_empty.push(vec![]);
            }
        }
        verify_large_roundtrip(&with_empty);
    }
}
