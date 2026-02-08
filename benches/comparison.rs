use criterion::{black_box, criterion_group, criterion_main, Criterion};
use csv::{ReaderBuilder, WriterBuilder};
use nsv::{encode, decode};

fn generate_test_data(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!("row{}_col{}", i, j))
                .collect()
        })
        .collect()
}

fn generate_test_data_with_multiline(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    if j % 3 == 0 {
                        format!("Line1\nLine2\nrow{}_col{}", i, j)
                    } else {
                        format!("row{}_col{}", i, j)
                    }
                })
                .collect()
        })
        .collect()
}

fn generate_test_data_with_commas(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!("item{}, variant{}, category{}", i, j, i + j))
                .collect()
        })
        .collect()
}

fn generate_test_data_with_quotes(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| format!(r#"He said "hello" at row {} col {}"#, i, j))
                .collect()
        })
        .collect()
}

fn generate_test_data_mixed_special(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    let patterns = [
        r#"John "Johnny" Doe, Jr."#,
        "Email: user@example.com",
        "Description:\nMulti-line\nText here",
        r#"Quote: "To be or not to be""#,
        r"Path: C:\Users\Documents",
        "Data: value1, value2, value3",
        "",
        "Normal text without special chars",
    ];

    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    let pattern_idx = (i * cells_per_row + j) % patterns.len();
                    format!("{} [r{}c{}]", patterns[pattern_idx], i, j)
                })
                .collect()
        })
        .collect()
}

fn generate_test_data_heavy_backslashes(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    // NSV worst case: lots of backslashes that need escaping
    let patterns = [
        r"C:\Windows\System32\drivers\etc\hosts",
        r"\\network\share\path\to\file.txt",
        r"Regex: \d+\.\d+\.\d+\.\d+",
        r"Escaped: \n\t\r\x00\u0000",
        r"UNC: \\?\C:\Very\Long\Path",
        "LaTeX: \\textbf{bold} \\textit{italic} \\\\",
    ];

    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    let pattern_idx = (i * cells_per_row + j) % patterns.len();
                    format!("{} [row{}]", patterns[pattern_idx], i)
                })
                .collect()
        })
        .collect()
}

fn generate_test_data_heavy_newlines(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    // NSV worst case: lots of newlines that need escaping
    let paragraph = "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed do eiusmod tempor incididunt\nut labore et dolore magna aliqua.\nUt enim ad minim veniam,\nquis nostrud exercitation ullamco\nlaboris nisi ut aliquip ex ea\ncommodo consequat.";

    let code_snippet = "fn main() {\n    println!(\"Hello\");\n    let x = 42;\n    if x > 0 {\n        println!(\"Positive\");\n    }\n}";

    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    if j % 2 == 0 {
                        format!("{} [{}]", paragraph, i)
                    } else {
                        format!("{} [{}]", code_snippet, i)
                    }
                })
                .collect()
        })
        .collect()
}

fn generate_realistic_heterogeneous_table(rows: usize) -> Vec<Vec<String>> {
    // Realistic table with different column types like a real CSV
    // Columns: ID, Name, Email, Date, Amount, Address, Status, Notes
    let first_names = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana"];
    let last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"];
    let domains = ["gmail.com", "yahoo.com", "company.com", "email.org"];
    let streets = ["Main St", "Oak Ave", "Pine Rd", "Elm Blvd"];
    let cities = ["Springfield", "Riverside", "Madison", "Georgetown"];
    let states = ["CA", "NY", "TX", "FL"];

    (0..rows)
        .map(|i| {
            let first = first_names[i % first_names.len()];
            let last = last_names[(i / 3) % last_names.len()];
            let domain = domains[i % domains.len()];
            let street_num = 100 + (i * 7) % 900;
            let street = streets[i % streets.len()];
            let city = cities[i % cities.len()];
            let state = states[i % states.len()];
            let amount = (i as f64 * 12.34) % 10000.0;
            let year = 2020 + (i % 5);
            let month = 1 + (i % 12);
            let day = 1 + (i % 28);

            vec![
                format!("{:06}", i),  // ID
                format!("{} {}", first, last),  // Name
                format!("{}.{}@{}", first.to_lowercase(), last.to_lowercase(), domain),  // Email
                format!("{:04}-{:02}-{:02}", year, month, day),  // Date
                format!("{:.2}", amount),  // Amount
                format!("{} {}, {}, {} {}", street_num, street, city, state, 10000 + (i % 90000)),  // Address with commas
                if i % 3 == 0 { "Active" } else if i % 3 == 1 { "Pending" } else { "Inactive" }.to_string(),  // Status
                if i % 5 == 0 {
                    format!("Customer since {}.\nPreferred contact: email.", year - 2)
                } else {
                    "".to_string()
                },  // Notes (sometimes multiline, often empty)
            ]
        })
        .collect()
}

fn generate_nested_encoding_data(rows: usize, cells_per_row: usize) -> Vec<Vec<String>> {
    // Both formats suffer: encoding structured data inside cells
    let json_samples = [
        r#"{"id": 1, "name": "Item A", "price": 9.99}"#,
        r#"{"tags": ["urgent", "review"], "count": 42}"#,
        r#"{"nested": {"deep": {"value": "test"}}}"#,
    ];

    let csv_samples = [
        "sub1,sub2,sub3",
        "\"quoted,value\",normal,\"another,quoted\"",
        "a,b,c,d,e,f",
    ];

    (0..rows)
        .map(|i| {
            (0..cells_per_row)
                .map(|j| {
                    if j % 2 == 0 {
                        json_samples[(i * cells_per_row + j) % json_samples.len()].to_string()
                    } else {
                        csv_samples[(i * cells_per_row + j) % csv_samples.len()].to_string()
                    }
                })
                .collect()
        })
        .collect()
}

fn data_to_csv(data: &[Vec<String>]) -> String {
    let mut wtr = WriterBuilder::new().from_writer(vec![]);
    for row in data {
        wtr.write_record(row).unwrap();
    }
    String::from_utf8(wtr.into_inner().unwrap()).unwrap()
}

fn csv_to_data(csv: &str) -> Vec<Vec<String>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(csv.as_bytes());
    rdr.records()
        .map(|r| r.unwrap().iter().map(|s| s.to_string()).collect())
        .collect()
}

fn bench_comparison_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_simple_10k");

    let data = generate_test_data(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Simple Data (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_large_100k");

    let data = generate_test_data(100_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Large Data (100K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_multiline(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_multiline_10k");

    let data = generate_test_data_with_multiline(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Multiline Data (10K rows x 10 cols with newlines) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_wide_1k");

    let data = generate_test_data(1_000, 100); // 1K rows, 100 columns
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Wide Data (1K rows x 100 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_with_commas(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_with_commas_10k");

    let data = generate_test_data_with_commas(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Data with Commas (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_with_quotes(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_with_quotes_10k");

    let data = generate_test_data_with_quotes(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Data with Quotes (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_mixed_special(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_mixed_special_10k");

    let data = generate_test_data_mixed_special(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Mixed Special Characters (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_heavy_backslashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_heavy_backslashes_10k");

    let data = generate_test_data_heavy_backslashes(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Heavy Backslashes - NSV Worst Case (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_heavy_newlines(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_heavy_newlines_10k");

    let data = generate_test_data_heavy_newlines(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Heavy Newlines - NSV Worst Case (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_realistic_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_realistic_heterogeneous_10k");

    let data = generate_realistic_heterogeneous_table(10_000);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Realistic Heterogeneous Table (10K rows x 8 varied cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

fn bench_comparison_nested_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_nested_encoding_10k");

    let data = generate_nested_encoding_data(10_000, 10);
    let nsv_str = encode(&data);
    let csv_str = data_to_csv(&data);

    println!("\n=== Nested Encoding - Both Formats Challenged (10K rows x 10 cols) ===");
    println!("NSV size: {} bytes", nsv_str.len());
    println!("CSV size: {} bytes", csv_str.len());
    println!("Size ratio (CSV/NSV): {:.2}x", csv_str.len() as f64 / nsv_str.len() as f64);

    group.bench_function("nsv_parse", |b| b.iter(|| decode(black_box(&nsv_str))));

    group.bench_function("csv_parse", |b| b.iter(|| csv_to_data(black_box(&csv_str))));

    group.finish();
}

criterion_group!(
    benches,
    bench_comparison_simple,
    bench_comparison_large,
    bench_comparison_multiline,
    bench_comparison_wide,
    bench_comparison_with_commas,
    bench_comparison_with_quotes,
    bench_comparison_mixed_special,
    bench_comparison_heavy_backslashes,
    bench_comparison_heavy_newlines,
    bench_comparison_realistic_table,
    bench_comparison_nested_encoding
);
criterion_main!(benches);
