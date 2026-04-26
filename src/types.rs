//! Per-cell parsing and column type inference for NSV.
//!
//! Surface:
//!
//! - parse a single cell into a typed value ([`Type::try_cast`]),
//! - validate a cell against a declared type ([`Type::accepts`]),
//! - infer a column type from a sample of cells ([`infer_column`]).
//!
//! Empty cells encode NULL in NSV; the typed parsers reject `""`, the
//! inference pass skips empty cells, and an all-NULL column resolves to
//! [`Type::Varchar`].
//!
//! Consumers that need their own conversion semantics (e.g. a DuckDB
//! extension applying DuckDB's BOOL spellings, a pandas reader applying
//! pandas' `na_values`) should bypass this module and parse cells
//! themselves. This module commits to one specific set of accepted
//! spellings; widening it to be everyone's-conversion-rules is not the
//! goal.
//!
//! ## Accepted spellings
//!
//! - **BigInt**: signed decimal integer fitting in `i64`. No leading `+`,
//!   no whitespace, no separators, no scientific notation.
//! - **Double**: decimal or scientific notation parseable as `f64`,
//!   including `inf`/`nan` spellings Rust's float parser accepts.
//! - **Date**: `YYYY-MM-DD` (RFC 3339 `full-date`), validated against
//!   the proleptic Gregorian calendar.
//! - **Timestamp**: `YYYY-MM-DD(T| )HH:MM:SS[.fraction](Z|±HH:MM)`
//!   (RFC 3339 `date-time`). Offset is required.
//!
//! ## Stability
//!
//! New module in a 0.0.x crate. The set of types and the accepted
//! spellings are not yet settled — both may change before 1.0.

// ── Type enum ────────────────────────────────────────────────────────

/// Column type produced by inference, or accepted by per-cell casts.
///
/// Variants are listed in the order [`DEFAULT_INFERENCE_ORDER`] tries
/// them. [`Type::Varchar`] always accepts and so sits last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    BigInt,
    Double,
    Date,
    Timestamp,
    Varchar,
}

impl Type {
    /// Parse a non-empty cell into a typed value, or return `None` on failure.
    ///
    /// Empty input is always rejected (NSV uses empty cells for NULL).
    pub fn try_cast<'a>(&self, s: &'a str) -> Option<Value<'a>> {
        if s.is_empty() {
            return None;
        }
        match self {
            Type::BigInt => try_parse_bigint(s).map(Value::BigInt),
            Type::Double => try_parse_double(s).map(Value::Double),
            Type::Date => try_parse_date(s).map(Value::Date),
            Type::Timestamp => try_parse_timestamp(s).map(Value::Timestamp),
            Type::Varchar => Some(Value::Varchar(s)),
        }
    }

    /// `true` if [`Type::try_cast`] would succeed.
    pub fn accepts(&self, s: &str) -> bool {
        self.try_cast(s).is_some()
    }
}

// ── Value enum ───────────────────────────────────────────────────────

/// Result of [`Type::try_cast`]. `Varchar` borrows from the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    BigInt(i64),
    Double(f64),
    Date(Date),
    Timestamp(Timestamp),
    Varchar(&'a str),
}

/// RFC 3339 `full-date`: `YYYY-MM-DD` in the proleptic Gregorian calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Date {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// RFC 3339 `date-time`: date + time-of-day + UTC offset.
///
/// `nanosecond` carries the fractional second (0 if absent).
/// `offset_minutes` is the offset from UTC in minutes (`-00:00` is preserved
/// as `Some(0)`, distinct from `+00:00`/`Z` only via the `unknown_offset` flag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timestamp {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub nanosecond: u32,
    pub offset_minutes: i16,
    /// `true` iff the original offset was the RFC 3339 §4.3 `-00:00`
    /// "unknown local offset" sentinel.
    pub unknown_offset: bool,
}

// ── Per-type parsers ─────────────────────────────────────────────────

/// Parse as `i64` via [`str::parse`].
pub fn try_parse_bigint(s: &str) -> Option<i64> {
    s.parse::<i64>().ok()
}

/// Parse as `f64` via [`str::parse`].
pub fn try_parse_double(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

/// Parse RFC 3339 `full-date`: `YYYY-MM-DD`.
pub fn try_parse_date(s: &str) -> Option<Date> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let year = parse_u32(&b[0..4])? as i32;
    let month = parse_u32(&b[5..7])? as u8;
    let day = parse_u32(&b[8..10])? as u8;
    if !is_valid_ymd(year, month, day) {
        return None;
    }
    Some(Date { year, month, day })
}

/// Parse RFC 3339 `date-time`: `YYYY-MM-DD(T| )HH:MM:SS[.fraction](Z|±HH:MM)`.
///
/// Per RFC 3339 §5.6, the date/time separator may be uppercase `T` or a
/// space; we accept both. The offset is required.
pub fn try_parse_timestamp(s: &str) -> Option<Timestamp> {
    let b = s.as_bytes();
    if b.len() < 20 {
        return None;
    }
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    if b[10] != b'T' && b[10] != b' ' {
        return None;
    }
    if b[13] != b':' || b[16] != b':' {
        return None;
    }

    let year = parse_u32(&b[0..4])? as i32;
    let month = parse_u32(&b[5..7])? as u8;
    let day = parse_u32(&b[8..10])? as u8;
    let hour = parse_u32(&b[11..13])? as u8;
    let minute = parse_u32(&b[14..16])? as u8;
    let second = parse_u32(&b[17..19])? as u8;

    if !is_valid_ymd(year, month, day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 60 {
        // RFC 3339 §5.6 allows second=60 for leap seconds.
        return None;
    }

    let mut i = 19;
    let mut nanosecond: u32 = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let frac = &b[frac_start..i];
        if frac.is_empty() {
            return None;
        }
        nanosecond = scale_fraction_to_nanos(frac)?;
    }

    let (offset_minutes, unknown_offset) = parse_offset(&b[i..])?;

    Some(Timestamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        offset_minutes,
        unknown_offset,
    })
}

fn parse_offset(b: &[u8]) -> Option<(i16, bool)> {
    if b == b"Z" {
        return Some((0, false));
    }
    if b.len() != 6 {
        return None;
    }
    let sign = match b[0] {
        b'+' => 1i16,
        b'-' => -1i16,
        _ => return None,
    };
    if b[3] != b':' {
        return None;
    }
    let hh = parse_u32(&b[1..3])?;
    let mm = parse_u32(&b[4..6])?;
    if hh > 23 || mm > 59 {
        return None;
    }
    let total = sign * (hh as i16 * 60 + mm as i16);
    let unknown = sign == -1 && hh == 0 && mm == 0;
    Some((total, unknown))
}

fn scale_fraction_to_nanos(digits: &[u8]) -> Option<u32> {
    // Use up to 9 fractional digits; truncate the rest. RFC 3339 doesn't
    // bound the fraction length, but anything past nanoseconds is dropped.
    let take = digits.len().min(9);
    let mut value: u32 = 0;
    for &d in &digits[..take] {
        if !d.is_ascii_digit() {
            return None;
        }
        value = value * 10 + (d - b'0') as u32;
    }
    for &d in &digits[take..] {
        if !d.is_ascii_digit() {
            return None;
        }
    }
    for _ in take..9 {
        value *= 10;
    }
    Some(value)
}

fn parse_u32(b: &[u8]) -> Option<u32> {
    let mut v: u32 = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v.checked_mul(10)?.checked_add((c - b'0') as u32)?;
    }
    Some(v)
}

fn is_valid_ymd(year: i32, month: u8, day: u8) -> bool {
    if !(1..=12).contains(&month) || day < 1 {
        return false;
    }
    let max = days_in_month(year, month);
    day <= max
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// ── Inference ────────────────────────────────────────────────────────

/// Order in which [`infer_column`] tries candidate types. The first
/// candidate that accepts every non-empty cell wins.
pub const DEFAULT_INFERENCE_ORDER: &[Type] = &[
    Type::BigInt,
    Type::Double,
    Type::Date,
    Type::Timestamp,
    Type::Varchar,
];

/// Recommended sample size for column inference.
pub const DEFAULT_SAMPLE_SIZE: usize = 1000;

/// Infer a column type from a sample of cells using [`DEFAULT_INFERENCE_ORDER`].
///
/// Empty cells are treated as NULL and skipped. An all-NULL sample
/// resolves to [`Type::Varchar`]. Callers should pre-truncate to
/// [`DEFAULT_SAMPLE_SIZE`] cells.
pub fn infer_column<I, S>(cells: I) -> Type
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    infer_column_with(cells, DEFAULT_INFERENCE_ORDER)
}

/// Infer a column type using a caller-supplied candidate order.
///
/// The first candidate that accepts every non-empty cell wins; if no
/// candidate accepts every cell, returns [`Type::Varchar`]. If `order`
/// is empty, returns [`Type::Varchar`].
pub fn infer_column_with<I, S>(cells: I, order: &[Type]) -> Type
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // Materialize once; we walk the sample per candidate.
    let sample: Vec<S> = cells.into_iter().collect();
    let non_empty: Vec<&str> = sample
        .iter()
        .map(|s| s.as_ref())
        .filter(|s| !s.is_empty())
        .collect();

    if non_empty.is_empty() {
        return Type::Varchar;
    }

    for &candidate in order {
        if candidate == Type::Varchar {
            return Type::Varchar;
        }
        if non_empty.iter().all(|s| candidate.accepts(s)) {
            return candidate;
        }
    }
    Type::Varchar
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // BigInt
    #[test]
    fn bigint_round_numbers() {
        assert_eq!(try_parse_bigint("0"), Some(0));
        assert_eq!(try_parse_bigint("-1"), Some(-1));
        assert_eq!(try_parse_bigint("9223372036854775807"), Some(i64::MAX));
        assert_eq!(try_parse_bigint("-9223372036854775808"), Some(i64::MIN));
    }

    #[test]
    fn bigint_rejects_overflow_and_junk() {
        assert_eq!(try_parse_bigint("9223372036854775808"), None);
        assert_eq!(try_parse_bigint("1.0"), None);
        assert_eq!(try_parse_bigint("1e3"), None);
        assert_eq!(try_parse_bigint(" 1"), None);
        assert_eq!(try_parse_bigint("1_000"), None);
        assert_eq!(try_parse_bigint(""), None);
    }

    // Double
    #[test]
    fn double_round_numbers() {
        assert_eq!(try_parse_double("0"), Some(0.0));
        assert_eq!(try_parse_double("2.5"), Some(2.5));
        assert_eq!(try_parse_double("-1e10"), Some(-1e10));
    }

    #[test]
    fn double_rejects_junk() {
        assert_eq!(try_parse_double(""), None);
        assert_eq!(try_parse_double("abc"), None);
        assert_eq!(try_parse_double("1,5"), None);
    }

    // Date
    #[test]
    fn date_strict_rfc3339() {
        let d = try_parse_date("2024-01-15").unwrap();
        assert_eq!((d.year, d.month, d.day), (2024, 1, 15));
        assert_eq!(try_parse_date("2024-02-29").map(|d| d.day), Some(29));
        assert_eq!(try_parse_date("2023-02-29"), None);
        assert_eq!(try_parse_date("2000-02-29").map(|d| d.day), Some(29));
        assert_eq!(try_parse_date("1900-02-29"), None);
    }

    #[test]
    fn date_rejects_alternate_formats() {
        assert_eq!(try_parse_date(""), None);
        assert_eq!(try_parse_date("2024-1-15"), None);
        assert_eq!(try_parse_date("2024/01/15"), None);
        assert_eq!(try_parse_date("24-01-15"), None);
        assert_eq!(try_parse_date("2024-13-01"), None);
        assert_eq!(try_parse_date("2024-04-31"), None);
        assert_eq!(try_parse_date("2024-01-15T"), None);
    }

    // Timestamp
    #[test]
    fn timestamp_strict_rfc3339() {
        let t = try_parse_timestamp("2024-01-15T12:34:56Z").unwrap();
        assert_eq!(t.hour, 12);
        assert_eq!(t.offset_minutes, 0);
        assert!(!t.unknown_offset);

        let t = try_parse_timestamp("2024-01-15T12:34:56.789+02:00").unwrap();
        assert_eq!(t.nanosecond, 789_000_000);
        assert_eq!(t.offset_minutes, 120);

        let t = try_parse_timestamp("2024-01-15 12:34:56-05:30").unwrap();
        assert_eq!(t.offset_minutes, -(5 * 60 + 30));

        let t = try_parse_timestamp("2024-01-15T12:34:56-00:00").unwrap();
        assert_eq!(t.offset_minutes, 0);
        assert!(t.unknown_offset);
    }

    #[test]
    fn timestamp_fractional_truncates_past_nanos() {
        let t = try_parse_timestamp("2024-01-15T00:00:00.1234567890123Z").unwrap();
        assert_eq!(t.nanosecond, 123_456_789);
    }

    #[test]
    fn timestamp_rejects_non_rfc3339() {
        assert_eq!(try_parse_timestamp(""), None);
        assert_eq!(try_parse_timestamp("2024-01-15T12:34:56"), None); // no offset
        assert_eq!(try_parse_timestamp("2024-01-15T12:34Z"), None);
        assert_eq!(try_parse_timestamp("2024-01-15T25:00:00Z"), None);
        assert_eq!(try_parse_timestamp("2024-01-15T12:34:56.Z"), None);
        assert_eq!(try_parse_timestamp("2024-01-15T12:34:56+25:00"), None);
    }

    // try_cast / accepts
    #[test]
    fn try_cast_dispatches_per_type() {
        assert!(matches!(Type::BigInt.try_cast("42"), Some(Value::BigInt(42))));
        assert!(matches!(Type::Double.try_cast("2.5"), Some(Value::Double(_))));
        assert!(matches!(Type::Varchar.try_cast("anything"), Some(Value::Varchar("anything"))));
        assert!(Type::BigInt.try_cast("").is_none());
        assert!(Type::Varchar.try_cast("").is_none());
    }

    #[test]
    fn accepts_matches_try_cast() {
        for s in ["true", "42", "3.14", "2024-01-01", "2024-01-01T00:00:00Z", "x", ""] {
            for t in DEFAULT_INFERENCE_ORDER {
                assert_eq!(t.accepts(s), t.try_cast(s).is_some(), "{:?} on {:?}", t, s);
            }
        }
    }

    // Inference
    #[test]
    fn infer_picks_most_specific() {
        assert_eq!(infer_column(["1", "2", "3"]), Type::BigInt);
        assert_eq!(infer_column(["1", "2.0", "3"]), Type::Double);
        assert_eq!(infer_column(["2024-01-01", "2024-12-31"]), Type::Date);
        assert_eq!(
            infer_column(["2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"]),
            Type::Timestamp
        );
        assert_eq!(infer_column(["true", "false"]), Type::Varchar);
        assert_eq!(infer_column(["a", "b"]), Type::Varchar);
    }

    #[test]
    fn infer_skips_empty_cells() {
        assert_eq!(infer_column(["1", "", "2", ""]), Type::BigInt);
        assert_eq!(infer_column(["", "", ""]), Type::Varchar);
        let empty: [&str; 0] = [];
        assert_eq!(infer_column(empty), Type::Varchar);
    }

    #[test]
    fn infer_falls_through_on_mixed() {
        assert_eq!(infer_column(["1", "abc"]), Type::Varchar);
        assert_eq!(infer_column(["1.0", "abc"]), Type::Varchar);
    }

    #[test]
    fn infer_with_custom_order() {
        // Skip BigInt: "42" should fall through to Double.
        let order = &[Type::Double, Type::Varchar];
        assert_eq!(infer_column_with(["42"], order), Type::Double);
        assert_eq!(infer_column_with(["42"], &[]), Type::Varchar);
    }
}
