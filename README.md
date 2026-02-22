# NSV Rust

Rust implementation of the [NSV (Newline-Separated Values)](https://nsv-format.org) format.

Parallel chunked parsing via `rayon` + `memchr`, byte-level API with no encoding assumptions, column-selective (projected) decode.

## Installation

```sh
cargo add nsv
```

## Usage

### Basic encoding/decoding

```rust
use nsv::{decode, encode};

let data = decode("a\nb\nc\n\nd\ne\nf\n\n");
// [["a", "b", "c"], ["d", "e", "f"]]

let encoded = encode(&data);
// "a\nb\nc\n\nd\ne\nf\n\n"
```

### Cell-level escaping

```rust
use nsv::{escape, unescape};

escape("hello\nworld");  // "hello\\nworld"
escape("");              // "\\"

unescape("hello\\nworld");  // "hello\nworld"
unescape("\\");              // ""
```

### Byte-level API

All core operations have `_bytes` variants for working with arbitrary ASCII-compatible encodings (Latin-1, Shift-JIS, raw binary, etc). No UTF-8 assumption.

```rust
use nsv::{decode_bytes, encode_bytes, escape_bytes, unescape_bytes};

let data = decode_bytes(b"a\nb\n\nc\nd\n\n");
let encoded = encode_bytes(&data);
```

### Projected decode

Column-selective parsing. Single-pass scan that tracks the column index, skips non-projected columns entirely (no allocation, no unescape), and produces the result directly.

```rust
use nsv::decode_bytes_projected;

let input = b"name\nage\nsalary\n\nAlice\n30\n50000\n\nBob\n25\n75000\n\n";

// Extract only columns 0 and 2 (name, salary)
let projected = decode_bytes_projected(input, &[0, 2]);
// [[b"name", b"salary"], [b"Alice", b"50000"], [b"Bob", b"75000"]]

// Reorder: columns appear in the order specified
let reordered = decode_bytes_projected(input, &[2, 0]);
// [[b"salary", b"name"], [b"50000", b"Alice"], [b"75000", b"Bob"]]
```

### Validation

```rust
use nsv::check;

let warnings = check(b"hello\\x\nworld\n");
for w in &warnings {
    println!("{}:{} {:?}", w.line, w.col, w.kind);
}
```

Warning kinds: `UnknownEscape(u8)`, `DanglingBackslash`, `NoTerminalLf`.

`check` is opt-in diagnostics — it doesn't alter parsing behavior.

### Structural operations (spill/unspill)

```rust
use nsv::util::{spill, unspill};

let flat = spill(&vec![vec!["a", "b"], vec!["c"]], "");
// ["a", "b", "", "c", ""]

let structured = unspill(&flat, &"");
// [["a", "b"], ["c"]]
```

Generic over `T: Clone + PartialEq` — works with strings, bytes, integers, anything.

### Composition

`nsv::util` also exposes the algebraic decomposition of encode/decode:

```
encode = spill('\n') ∘ spill("") ∘ escape_seqseq
decode = unescape_seqseq ∘ unspill("") ∘ unspill('\n')
```

```rust
use nsv::util::{escape_seqseq, unescape_seqseq, spill, unspill};

let escaped = escape_seqseq(&data);
let flat = spill(&escaped, String::new());
let chars: Vec<char> = spill(
    &flat.iter().map(|s| s.chars().collect()).collect::<Vec<Vec<char>>>(),
    '\n',
);
let encoded: String = chars.into_iter().collect();
assert_eq!(nsv::encode(&data), encoded);
```

## API

### Core

| Function | Signature |
|----------|-----------|
| `decode` | `(&str) -> Vec<Vec<String>>` |
| `encode` | `(&[Vec<String>]) -> String` |
| `decode_bytes` | `(&[u8]) -> Vec<Vec<Vec<u8>>>` |
| `encode_bytes` | `(&[Vec<Vec<u8>>]) -> Vec<u8>` |
| `decode_bytes_projected` | `(&[u8], &[usize]) -> Vec<Vec<Vec<u8>>>` |

### Cell escaping

| Function | Signature |
|----------|-----------|
| `escape` / `unescape` | `(&str) -> String` |
| `escape_bytes` / `unescape_bytes` | `(&[u8]) -> Vec<u8>` |

### Validation

| Function | Signature |
|----------|-----------|
| `check` | `(&[u8]) -> Vec<Warning>` |

### Util (`nsv::util`)

| Function | Description |
|----------|-------------|
| `spill` / `unspill` | Flatten/recover seqseq dimension with terminators |
| `escape_seqseq` / `unescape_seqseq` | `map(map(escape))` / `map(map(unescape))` over a seqseq |

## Parallel parsing

For inputs above 64KB, `decode_bytes` (and `decode`, and `decode_bytes_projected`) switch from sequential to chunked parallel parsing:

1. Pick N evenly-spaced byte positions (one per CPU core)
2. Scan forward from each to the nearest `\n\n` row boundary — O(avg_row_len)
3. Each worker independently parses its chunk (boundary scan + cell split + unescape)

This works because literal `0x0A` in NSV is always structural (never escaped), so row-boundary recovery from any byte position is a trivial forward scan. The sequential phase is O(N), not O(input_len) — all real work is parallel.

