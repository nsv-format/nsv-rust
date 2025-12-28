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

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(seqseq: &[Vec<String>]) -> String {
        let escaped = escape_seqseq(seqseq);
        let spilled = spill(&escaped, String::new());
        spilled.iter().flat_map(|s| s.chars().chain(std::iter::once('\n'))).collect()
    }

    fn decode(s: &str) -> Vec<Vec<String>> {
        let mut strings: Vec<String> = s.split('\n').map(|s| s.to_string()).collect();
        strings.pop(); // Remove trailing element from split
        unescape_seqseq(&unspill(&strings, &String::new()))
    }

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

        let ints: Vec<Vec<i32>> = vec![vec![1, 2], vec![3]];
        assert_eq!(spill(&ints, -1), vec![1, 2, -1, 3, -1]);
    }

    #[test]
    fn test_unspill_discards_incomplete() {
        let input = vec!["a".into(), "b".into()];
        assert!(unspill(&input, &String::new()).is_empty());
    }

    #[test]
    fn test_decomposition_matches_direct() {
        let samples: Vec<Vec<Vec<String>>> = vec![
            vec![],
            vec![vec![]],
            vec![vec![], vec![], vec![]],
            vec![vec!["a".into(), "b".into()], vec!["c".into(), "d".into()]],
            vec![vec!["".into(), "x".into(), "".into()]],
            vec![vec!["field\\with\\backslashes".into(), "field\nwith\nnewlines".into()]],
            vec![vec!["\\n".into(), "\\\n".into()], vec!["\\\\".into(), "\n\n".into()]],
        ];
        for seqseq in samples {
            assert_eq!(crate::dumps(&seqseq), encode(&seqseq));
            assert_eq!(seqseq, decode(&encode(&seqseq)));
        }
    }

    #[test]
    fn test_large_datasets() {
        let verify = |original: &[Vec<String>]| {
            let direct = crate::dumps(original);
            assert!(direct.len() > 64 * 1024);
            assert_eq!(direct, encode(original));
            assert_eq!(original, decode(&direct));
            assert_eq!(crate::loads(&direct), decode(&direct));
        };

        let simple: Vec<Vec<String>> = (0..10_000)
            .map(|i| vec![format!("row{}", i), format!("data{}", i)])
            .collect();
        verify(&simple);

        let with_escapes: Vec<Vec<String>> = (0..10_000)
            .map(|i| vec![format!("line\n{}", i), format!("back\\{}", i), "".into()])
            .collect();
        verify(&with_escapes);

        let mut with_empty = Vec::new();
        for i in 0..10_000 {
            with_empty.push(vec![format!("v{}", i)]);
            if i % 100 == 0 { with_empty.push(vec![]); }
        }
        verify(&with_empty);
    }
}
