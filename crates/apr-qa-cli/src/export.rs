
/// Auto-populate model cache directory with symlinks from pacha and HF caches.
///
/// Creates `gguf/`, `apr/`, `safetensors/` subdirectories and symlinks model files
/// from the pacha cache (`~/.cache/pacha/models/`) and HuggingFace cache
/// (`~/.cache/huggingface/hub/`). The `apr/` subdirectory is populated during
/// 6-column profiling (GGUF → APR conversion).
fn auto_populate_model_cache(model_id: &str, model_dir: &std::path::Path, apr_binary: &str) {
    let gguf_dir = model_dir.join("gguf");
    let apr_dir = model_dir.join("apr");
    let st_dir = model_dir.join("safetensors");

    if gguf_dir.exists() && has_file_with_ext(&gguf_dir, "gguf") {
        println!("  Cache already populated: {}", model_dir.display());
        return;
    }

    println!("  Auto-populating model cache...");

    for dir in [&gguf_dir, &apr_dir, &st_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("  Error creating {}: {e}", dir.display());
            return;
        }
    }

    run_apr_pull(apr_binary, model_id);

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let home = std::path::Path::new(&home);

    link_gguf_from_pacha(model_id, home, &gguf_dir);
    link_safetensors_from_hf(model_id, home, &st_dir);
}

fn run_apr_pull(apr_binary: &str, model_id: &str) {
    println!("  Running: {apr_binary} pull {model_id}");
    let pull_status = std::process::Command::new(apr_binary)
        .args(["pull", model_id])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match pull_status {
        Ok(s) if s.success() => println!("  Pull succeeded"),
        Ok(s) => eprintln!("  Pull exited with: {s}"),
        Err(e) => eprintln!("  Pull failed: {e}"),
    }
}

fn link_gguf_from_pacha(model_id: &str, home: &std::path::Path, gguf_dir: &std::path::Path) {
    let manifest_path = home.join(".cache/pacha/models/manifest.json");
    let Some(gguf_path) = find_gguf_in_pacha(&manifest_path, model_id) else {
        eprintln!("  No GGUF found in pacha cache for {model_id}");
        return;
    };
    let link = gguf_dir.join("model.gguf");
    if link.exists() {
        return;
    }
    match std::os::unix::fs::symlink(&gguf_path, &link) {
        Ok(()) => println!("  Linked GGUF: {gguf_path}"),
        Err(e) => eprintln!("  Error symlinking GGUF: {e}"),
    }
}

fn link_safetensors_from_hf(model_id: &str, home: &std::path::Path, st_dir: &std::path::Path) {
    let (org, repo) = split_model_id(model_id);
    let hf_model_dir = home
        .join(".cache/huggingface/hub")
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");

    let Some(st_path) = find_safetensors_in_hf(&hf_model_dir) else {
        eprintln!("  No SafeTensors found in HF cache for {model_id}");
        return;
    };

    let link = st_dir.join("model.safetensors");
    if !link.exists() {
        match std::os::unix::fs::symlink(&st_path, &link) {
            Ok(()) => println!("  Linked SafeTensors: {}", st_path.display()),
            Err(e) => eprintln!("  Error symlinking SafeTensors: {e}"),
        }
    }

    // Copy config.json from the same snapshot directory
    let Some(snapshot_dir) = st_path.parent() else {
        return;
    };
    let config_src = snapshot_dir.join("config.json");
    let config_dst = st_dir.join("config.json");
    if config_src.exists() && !config_dst.exists() {
        match std::fs::copy(&config_src, &config_dst) {
            Ok(_) => println!("  Copied config.json"),
            Err(e) => eprintln!("  Error copying config.json: {e}"),
        }
    }
}

/// Check if a directory contains a file with the given extension.
fn has_file_with_ext(dir: &std::path::Path, ext: &str) -> bool {
    dir.read_dir()
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == ext))
        })
        .unwrap_or(false)
}

/// Find a GGUF file in the pacha cache manifest matching the model ID.
///
/// Pacha manifest entries use the naming convention:
/// `hf_Org_Repo-GGUF_repo-name-q4_k_m.gguf`
fn find_gguf_in_pacha(manifest_path: &std::path::Path, model_id: &str) -> Option<String> {
    let content = std::fs::read_to_string(manifest_path).ok()?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&content).ok()?;

    // Build search key from model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct" → "Qwen_Qwen2.5-Coder-1.5B-Instruct"
    let (org, repo) = split_model_id(model_id);
    let gguf_key = format!("hf_{org}_{repo}-GGUF_");

    // Find first GGUF entry matching this model
    for entry in &entries {
        let name = entry.get("name")?.as_str()?;
        if name.starts_with(&gguf_key)
            && std::path::Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return entry.get("path")?.as_str().map(String::from);
        }
    }

    None
}

/// Find a `model.safetensors` file in the HuggingFace cache snapshots directory.
fn find_safetensors_in_hf(snapshots_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(snapshots_dir).ok()?;
    for entry in entries.flatten() {
        let snapshot = entry.path();
        if snapshot.is_dir() {
            let st_file = snapshot.join("model.safetensors");
            if st_file.exists() {
                return Some(st_file);
            }
        }
    }
    None
}

/// Split a HuggingFace model ID into (org, repo).
///
/// e.g. `"Qwen/Qwen2.5-Coder-1.5B-Instruct"` → `("Qwen", "Qwen2.5-Coder-1.5B-Instruct")`
fn split_model_id(model_id: &str) -> (&str, &str) {
    model_id.split_once('/').unwrap_or(("unknown", model_id))
}

/// Export certification data to models.csv (PMAT-264)
///
/// Scans evidence directory, calculates MQS for each evidence file,
/// and writes/updates models.csv for oracle consumption.
#[allow(clippy::too_many_lines)]
fn export_csv(evidence_dir: &Path, output: &Path, append: bool) {
    use apr_qa_report::write_models_csv;

    println!("Exporting certification data to CSV...");
    println!("  Evidence directory: {}", evidence_dir.display());
    println!("  Output: {}", output.display());
    println!("  Mode: {}", if append { "append" } else { "overwrite" });

    let mut rows = load_existing_csv_rows(output, append);
    let (processed, updated) = process_evidence_files(evidence_dir, &mut rows);

    if processed == 0 {
        println!("  No evidence files found in {}", evidence_dir.display());
        return;
    }

    ensure_parent_dir(output);
    if let Err(e) = write_models_csv(&rows, output) {
        eprintln!("Error: Failed to write CSV: {e}");
        std::process::exit(1);
    }

    println!("\nExported {} row(s) to {}", rows.len(), output.display());
    println!("  Processed: {processed}");
    println!("  Updated: {updated}");
    println!("  New: {}", processed - updated);
}

fn load_existing_csv_rows(output: &Path, append: bool) -> Vec<apr_qa_report::CertificationRow> {
    use apr_qa_report::read_models_csv;

    if !append || !output.exists() {
        return Vec::new();
    }
    match read_models_csv(output) {
        Ok(existing) => {
            println!("  Loaded {} existing row(s)", existing.len());
            existing
        }
        Err(e) => {
            eprintln!("Warning: Could not read existing CSV: {e}");
            Vec::new()
        }
    }
}

fn process_evidence_files(
    evidence_dir: &Path,
    rows: &mut Vec<apr_qa_report::CertificationRow>,
) -> (usize, usize) {
    let entries = match std::fs::read_dir(evidence_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: Cannot read evidence directory: {e}");
            std::process::exit(1);
        }
    };

    let mut processed = 0;
    let mut updated = 0;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Warning: Could not read {}: {e}", path.display());
                continue;
            }
        };
        let Ok(export) = serde_json::from_str::<apr_qa_report::EvidenceExport>(&content) else {
            continue;
        };
        processed += 1;
        let was_updated = update_row_from_export(rows, &export);
        if was_updated {
            updated += 1;
        }
    }
    (processed, updated)
}

#[allow(clippy::option_if_let_else, clippy::single_match_else)]
fn update_row_from_export(
    rows: &mut Vec<apr_qa_report::CertificationRow>,
    export: &apr_qa_report::EvidenceExport,
) -> bool {
    use apr_qa_report::CertificationRow;
    use chrono::Utc;

    let model_id = &export.model.hf_repo;
    // Can't use map_or_else here due to borrow checker - we need mutable access to rows
    let (row_idx, was_updated) = match rows.iter().position(|r| r.model_id == *model_id) {
        Some(idx) => (idx, true),
        None => {
            rows.push(CertificationRow::new(model_id, &export.model.family));
            (rows.len() - 1, false)
        }
    };

    let row = &mut rows[row_idx];
    row.parameters.clone_from(&export.model.size);
    row.mqs_score = export.mqs.score;
    row.grade.clone_from(&export.mqs.grade);
    row.certified_tier.clone_from(&export.playbook.tier);
    row.last_certified = Utc::now();
    row.status = derive_status_from_mqs(&export.mqs);
    update_gateway_flags(row, &export.gates);

    println!(
        "  Processed: {} → MQS {}, {}",
        model_id, row.mqs_score, row.status
    );
    was_updated
}

#[allow(clippy::missing_const_for_fn)] // Can't be const due to internal use statement
fn derive_status_from_mqs(mqs: &apr_qa_report::MqsExport) -> apr_qa_report::ModelStatus {
    use apr_qa_report::ModelStatus;

    if mqs.score >= 800 && mqs.gateway_passed {
        ModelStatus::Certified
    } else if mqs.score == 0 {
        ModelStatus::Pending
    } else {
        ModelStatus::Blocked
    }
}

fn update_gateway_flags(
    row: &mut apr_qa_report::CertificationRow,
    gates: &std::collections::HashMap<String, apr_qa_report::GateResult>,
) {
    if let Some(g1) = gates.get("G1-MODEL-LOADS") {
        row.g1 = g1.passed;
    }
    if let Some(g2) = gates.get("G2-BASIC-INFERENCE") {
        row.g2 = g2.passed;
    }
    if let Some(g3) = gates.get("G3-NO-CRASHES") {
        row.g3 = g3.passed;
    }
    if let Some(g4) = gates.get("G4-OUTPUT-QUALITY") {
        row.g4 = g4.passed;
    }
}

fn ensure_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Error: Cannot create output directory: {e}");
            std::process::exit(1);
        }
    }
}

/// Export evidence to schema-compliant JSON (PMAT-265)
///
/// Converts test run results to the EvidenceExport format for oracle consumption.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn export_evidence(
    source: &Path,
    output_dir: &Path,
    model: &str,
    family: &str,
    size: &str,
    playbook_name: &str,
    tier: &str,
) {
    use apr_qa_report::{
        EvidenceExport, ExportSummary, GateResult, ModelMeta, MqsExport, PlaybookMeta,
    };
    use chrono::Utc;
    use std::collections::HashMap;

    println!("Exporting evidence to schema-compliant JSON...");
    println!("  Source: {}", source.display());
    println!("  Output dir: {}", output_dir.display());
    println!("  Model: {model}");

    // Read source file
    let content = match std::fs::read_to_string(source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Cannot read source file: {e}");
            std::process::exit(1);
        }
    };

    // Try to parse as execution result (from apr-qa run output)
    let json_value: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: Invalid JSON in source file: {e}");
            std::process::exit(1);
        }
    };

    // Extract evidence array and summary from execution result
    let evidence_array = json_value
        .get("evidence")
        .and_then(|e| e.as_array())
        .cloned()
        .unwrap_or_default();

    #[allow(clippy::cast_possible_truncation)]
    let total_scenarios = json_value
        .get("total_scenarios")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(evidence_array.len() as u64) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let passed = json_value
        .get("passed")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let failed = json_value
        .get("failed")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    #[allow(clippy::cast_possible_truncation)]
    let skipped = json_value
        .get("skipped")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    let duration_ms = json_value
        .get("duration_ms")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    #[allow(clippy::cast_precision_loss)]
    let pass_rate = if total_scenarios > 0 {
        passed as f64 / total_scenarios as f64
    } else {
        0.0
    };

    // Calculate MQS from pass rate (simplified)
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mqs_score = (pass_rate * 1000.0) as u32;
    let grade = match mqs_score {
        900..=1000 => "A",
        800..=899 => "B",
        600..=799 => "C",
        400..=599 => "D",
        _ => "F",
    };

    // Extract gateway results from evidence
    let mut gates: HashMap<String, GateResult> = HashMap::new();
    for ev in &evidence_array {
        if let Some(gate_id) = ev.get("gate_id").and_then(|g| g.as_str()) {
            if gate_id.starts_with('G') && !gates.contains_key(gate_id) {
                let passed = ev
                    .get("outcome")
                    .and_then(|o| o.as_str())
                    .is_some_and(|o| o == "Corroborated" || o == "Skipped");
                let reason = ev
                    .get("reason")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string();
                gates.insert(gate_id.to_string(), GateResult { passed, reason });
            }
        }
    }

    // Check if all gateways passed
    let gateway_passed = ["G1", "G2", "G3", "G4"].iter().all(|g| {
        gates
            .iter()
            .filter(|(k, _)| k.starts_with(g))
            .all(|(_, v)| v.passed)
    });

    // Build export structure
    let export = EvidenceExport {
        schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
        version: "1.0.0".to_string(),
        model: ModelMeta {
            hf_repo: model.to_string(),
            family: family.to_string(),
            size: size.to_string(),
            format: "safetensors".to_string(),
        },
        playbook: PlaybookMeta {
            name: playbook_name.to_string(),
            version: "1.0.0".to_string(),
            tier: tier.to_string(),
        },
        summary: ExportSummary {
            total_scenarios,
            passed,
            failed,
            skipped,
            pass_rate,
            duration_ms,
            timestamp: Utc::now(),
        },
        mqs: MqsExport {
            score: mqs_score,
            grade: grade.to_string(),
            gateway_passed,
            category_scores: HashMap::new(),
        },
        gates,
        evidence: evidence_array,
    };

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error: Cannot create output directory: {e}");
        std::process::exit(1);
    }

    // Generate output filename from model
    let safe_name = model.replace('/', "-").to_lowercase();
    let output_path = output_dir.join(format!("{safe_name}.json"));

    // Write export
    let json = match export.to_json() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Error: Failed to serialize export: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::write(&output_path, &json) {
        eprintln!("Error: Failed to write output: {e}");
        std::process::exit(1);
    }

    println!("\nExported evidence to: {}", output_path.display());
    println!("  Model: {model}");
    println!("  MQS Score: {mqs_score}");
    println!("  Grade: {grade}");
    println!("  Pass Rate: {:.1}%", pass_rate * 100.0);
    println!("  Total Scenarios: {total_scenarios}");
}

/// Validate a model against the tensor layout contract (Issue #4)
///
/// Checks that tensor shapes in the APR model match the contract expectations.
/// This prevents GH-202 style bugs where wrong shapes cause garbage output.
fn validate_contract_command(
    model_path: &Path,
    contract_path: Option<&Path>,
    format: &str,
    critical_only: bool,
) {
    use apr_qa_runner::{get_critical_tensors, get_validation_rules, validate_model};

    println!("Validating model against tensor layout contract...");
    println!("  Model: {}", model_path.display());

    let contract = load_layout_contract(contract_path);
    println!("  Contract version: {}", contract.metadata.version);

    print_validation_rules(get_validation_rules(&contract));

    if critical_only {
        print_critical_tensors(get_critical_tensors(&contract));
    }

    println!("\n=== Running Validation ===");
    let result = match validate_model(model_path, &contract) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Validation failed: {e}");
            std::process::exit(1);
        }
    };

    output_validation_result(&result, format);
    exit_with_validation_status(result.passed);
}

/// Load the tensor layout contract, exiting on failure.
fn load_layout_contract(contract_path: Option<&Path>) -> apr_qa_runner::TensorLayoutContract {
    use apr_qa_runner::{load_contract, load_contract_from};

    let contract = contract_path.map_or_else(
        || {
            println!(
                "  Contract: {} (default)",
                apr_qa_runner::layout_contract::DEFAULT_CONTRACT_PATH
            );
            load_contract()
        },
        |path| {
            println!("  Contract: {}", path.display());
            load_contract_from(path)
        },
    );

    match contract {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Failed to load contract: {e}");
            eprintln!("\nHint: Ensure aprender is cloned as a sibling directory:");
            eprintln!("  ../aprender/contracts/tensor-layout-v1.yaml");
            std::process::exit(1);
        }
    }
}

/// Print validation rules from the contract.
fn print_validation_rules(rules: &[apr_qa_runner::ValidationRule]) {
    println!("\n=== Validation Rules ({}) ===", rules.len());
    for rule in rules {
        let critical_marker = if rule.critical { " [CRITICAL]" } else { "" };
        println!("  {}: {}{}", rule.id, rule.name, critical_marker);
    }
}

/// Print critical tensors from the contract.
fn print_critical_tensors(tensors: Vec<&apr_qa_runner::TensorSpec>) {
    println!("\n=== Critical Tensors ({}) ===", tensors.len());
    for tensor in &tensors {
        println!(
            "  {} -> {} (transpose: {})",
            tensor.gguf_name, tensor.apr_name, tensor.transpose
        );
    }
}

/// Output validation result in text or JSON format.
fn output_validation_result(result: &apr_qa_runner::ModelValidationResult, format: &str) {
    if format == "json" {
        match serde_json::to_string_pretty(result) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("Error serializing result: {e}"),
        }
        return;
    }

    println!("\n=== Validation Results ===");
    println!(
        "  Status: {}",
        if result.passed { "PASSED" } else { "FAILED" }
    );
    println!("  Rules Checked: {}", result.rules_checked);
    println!("  Rules Passed: {}", result.rules_passed);
    println!("  Rules Failed: {}", result.rules_failed);

    print_tensor_results(&result.tensor_results);
    print_critical_failures(&result.critical_failures);
}

/// Print per-tensor validation results.
fn print_tensor_results(tensor_results: &[apr_qa_runner::TensorValidationResult]) {
    if tensor_results.is_empty() {
        return;
    }
    println!("\n  Per-Tensor Results:");
    for tr in tensor_results {
        let status = if tr.passed { "✓" } else { "✗" };
        println!("    {} [{}] {}", status, tr.rule_id, tr.tensor_name);
        if !tr.passed {
            println!("      Details: {}", tr.details);
            if let Some(ref expected) = tr.expected {
                println!("      Expected: {expected}");
            }
            if let Some(ref actual) = tr.actual {
                println!("      Actual: {actual}");
            }
        }
    }
}

/// Print critical failures if any.
fn print_critical_failures(failures: &[String]) {
    if failures.is_empty() {
        return;
    }
    println!("\n  CRITICAL FAILURES:");
    for failure in failures {
        println!("    ✗ {failure}");
    }
}

/// Exit with appropriate status code based on validation result.
fn exit_with_validation_status(passed: bool) -> ! {
    if passed {
        println!("\n✓ Model conforms to tensor layout contract");
        std::process::exit(0);
    } else {
        println!("\n✗ Model DOES NOT conform to tensor layout contract");
        std::process::exit(1);
    }
}

/// Bootstrap an architecture-aware playbook from a family contract.
fn run_bootstrap(
    family: &str,
    size: &str,
    hf_repo: &str,
    tier: &str,
    output: Option<&Path>,
    contracts_path: &Path,
    dry_run: bool,
) {
    println!(
        "{} {}",
        "Bootstrapping playbook:".bold().cyan(),
        format!("{family}-{size}-{tier}").bold()
    );
    println!("  {} {hf_repo}", "HF Repo:".dimmed());
    println!("  {} {}", "Contracts:".dimmed(), contracts_path.display());

    match bootstrap_playbook_from_contract(family, size, hf_repo, tier, contracts_path) {
        Ok(yaml) => {
            if dry_run {
                println!("\n{yaml}");
            } else {
                let out_path = output.map_or_else(
                    || {
                        PathBuf::from(format!(
                            "playbooks/models/{family}-{size}-{tier}.playbook.yaml"
                        ))
                    },
                    PathBuf::from,
                );
                if let Some(parent) = out_path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("Error creating directory: {e}");
                        std::process::exit(1);
                    }
                }
                match std::fs::write(&out_path, &yaml) {
                    Ok(()) => {
                        println!("\n{} {}", "Written:".bold().green(), out_path.display());
                    }
                    Err(e) => {
                        eprintln!("Error writing playbook: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("{} {e}", "Error:".bold().red());
            std::process::exit(1);
        }
    }
}
