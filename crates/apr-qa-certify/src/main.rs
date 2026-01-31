//! Update README.md certification table from models.csv.
//!
//! Usage: apr-qa-readme-sync [--csv PATH] [--readme PATH]

#![forbid(unsafe_code)]

use apr_qa_certify::{
    CertifyError, END_MARKER, START_MARKER, generate_summary, generate_table, parse_csv,
    update_readme,
};
use chrono::Utc;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn find_project_root() -> Option<PathBuf> {
    let mut current = env::current_dir().ok()?;
    loop {
        if current.join("Cargo.toml").exists() && current.join("crates").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn run() -> Result<(), CertifyError> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut csv_path: Option<PathBuf> = None;
    let mut readme_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" => {
                i += 1;
                if i < args.len() {
                    csv_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--readme" => {
                i += 1;
                if i < args.len() {
                    readme_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: apr-qa-readme-sync [--csv PATH] [--readme PATH]");
                eprintln!();
                eprintln!("Updates README.md certification table from models.csv");
                eprintln!();
                eprintln!("Options:");
                eprintln!(
                    "  --csv PATH     Path to models.csv (default: docs/certifications/models.csv)"
                );
                eprintln!("  --readme PATH  Path to README.md (default: README.md)");
                eprintln!("  --help, -h     Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // Find project root
    let root = find_project_root().ok_or_else(|| {
        CertifyError::MarkerNotFound(
            "Could not find project root (looking for Cargo.toml + crates/)".to_string(),
        )
    })?;

    let csv_path = csv_path.unwrap_or_else(|| root.join("docs/certifications/models.csv"));
    let readme_path = readme_path.unwrap_or_else(|| root.join("README.md"));

    // Read CSV
    eprintln!("Reading CSV from: {}", csv_path.display());
    let csv_content = fs::read_to_string(&csv_path)?;
    let models = parse_csv(&csv_content)?;
    eprintln!("Loaded {} models", models.len());

    // Generate content
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let summary = generate_summary(&models, &timestamp);
    let table = generate_table(&models);
    let full_content = format!("{summary}\n\n{table}");

    // Read and update README
    eprintln!("Reading README from: {}", readme_path.display());
    let readme_content = fs::read_to_string(&readme_path)?;

    // Validate markers exist
    if !readme_content.contains(START_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing start marker: {START_MARKER}"
        )));
    }
    if !readme_content.contains(END_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing end marker: {END_MARKER}"
        )));
    }

    let updated_readme = update_readme(&readme_content, &full_content)?;

    // Write updated README
    fs::write(&readme_path, updated_readme)?;
    eprintln!("Updated {}", readme_path.display());
    eprintln!("Done. Commit both README.md and docs/certifications/models.csv together.");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
