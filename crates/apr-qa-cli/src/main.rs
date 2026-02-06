//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

use apr_qa_cli::{
    CertTier, PlaybookRunConfig, build_certification_config_with_policy, build_execution_config,
    calculate_mqs_score, calculate_popperian_score, collect_evidence, execute_auto_tickets,
    execute_playbook, filter_models_by_size, generate_html_report, generate_junit_report,
    generate_lock_file, generate_model_scenarios, generate_tickets_from_evidence, list_all_models,
    load_playbook, parse_evidence, parse_failure_policy, playbook_path_for_model,
    scenarios_to_json, scenarios_to_yaml,
};
use apr_qa_report::{MqsScore, PopperianScore};
use apr_qa_runner::ToolExecutor;
use apr_qa_runner::{Evidence, EvidenceCollector};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "apr-qa")]
#[command(about = "APR Model QA Playbook Runner", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// CertTier enum now comes from apr_qa_cli library

#[derive(Subcommand)]
enum Commands {
    /// Certify models against the verification matrix
    Certify {
        /// Certify all models in registry
        #[arg(long)]
        all: bool,

        /// Certify by model family (e.g., "qwen-coder", "llama")
        #[arg(long)]
        family: Option<String>,

        /// Certification tier (smoke, quick, standard, deep)
        #[arg(long, default_value = "quick")]
        tier: String,

        /// Specific model IDs to certify
        #[arg(value_name = "MODEL")]
        models: Vec<String>,

        /// Output directory for certification artifacts
        #[arg(short, long, default_value = "certifications")]
        output: PathBuf,

        /// Dry run (show what would be certified without running)
        #[arg(long)]
        dry_run: bool,

        /// Model cache directory (contains GGUF/APR/SafeTensors files)
        /// Structure: <cache>/<model-name>/<format>/<file>
        #[arg(long)]
        model_cache: Option<PathBuf>,

        /// Path to apr binary for real inference
        #[arg(long, default_value = "apr")]
        apr_binary: String,

        /// Auto-generate structured tickets from failures (§3.6)
        #[arg(long)]
        auto_ticket: bool,

        /// Repository for auto-ticket creation (e.g., "paiml/aprender")
        #[arg(long, default_value = "paiml/aprender")]
        ticket_repo: String,

        /// Disable playbook integrity checks (§3.1)
        #[arg(long)]
        no_integrity_check: bool,

        /// Stop on first failure with enhanced diagnostics (§12.5.3)
        #[arg(long)]
        fail_fast: bool,

        /// Enhance failures with batuta oracle context (§12.1.1)
        /// Generates falsification checklists and enriched metrics
        #[arg(long)]
        oracle_enhance: bool,
    },

    /// Run a playbook
    Run {
        /// Path to playbook YAML file
        #[arg(value_name = "PLAYBOOK")]
        playbook: PathBuf,

        /// Output directory for reports
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Failure policy (stop-on-first, stop-on-p0, collect-all, fail-fast)
        #[arg(long, default_value = "stop-on-p0")]
        failure_policy: String,

        /// Stop on first failure with enhanced diagnostics (§12.5.3)
        /// Equivalent to --failure-policy fail-fast
        /// Emits comprehensive trace output for debugging and GitHub ticket creation
        #[arg(long)]
        fail_fast: bool,

        /// Dry run (don't execute, just show what would be done)
        #[arg(long)]
        dry_run: bool,

        /// Maximum parallel workers
        #[arg(long, default_value = "4")]
        workers: usize,

        /// Path to model file
        #[arg(long)]
        model_path: Option<String>,

        /// Timeout per test in milliseconds
        #[arg(long, default_value = "60000")]
        timeout: u64,

        /// Disable GPU acceleration (use CPU only)
        #[arg(long)]
        no_gpu: bool,

        /// Skip P0 format conversion tests (NOT RECOMMENDED - these are critical)
        #[arg(long)]
        skip_conversion_tests: bool,

        /// Run APR tool coverage tests (inspect, validate, bench, check, trace, profile)
        #[arg(long)]
        run_tool_tests: bool,

        /// Run profile CI assertions (throughput, latency thresholds)
        #[arg(long)]
        profile_ci: bool,

        /// Skip differential tests (tensor_diff, inference_compare)
        #[arg(long)]
        no_differential: bool,

        /// Skip trace payload tests (forward pass, garbage detection)
        #[arg(long)]
        no_trace_payload: bool,

        /// Enable HF parity verification against golden corpus
        #[arg(long)]
        hf_parity: bool,

        /// Path to HF golden corpus directory
        #[arg(long, default_value = "../hf-ground-truth-corpus/oracle")]
        hf_corpus_path: String,

        /// HF parity model family (e.g., "qwen2.5-coder-1.5b/v1")
        #[arg(long)]
        hf_model_family: Option<String>,

        /// Disable playbook integrity checks (§3.1)
        #[arg(long)]
        no_integrity_check: bool,
    },

    /// Run APR tool coverage tests
    Tools {
        /// Path to model file
        #[arg(value_name = "MODEL_PATH")]
        model_path: PathBuf,

        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,

        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Include serve lifecycle test (F-INTEG-003)
        #[arg(long)]
        include_serve: bool,
    },

    /// Generate scenarios for a model
    Generate {
        /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        #[arg(value_name = "MODEL")]
        model: String,

        /// Number of scenarios per combination
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Output format (yaml, json)
        #[arg(short, long, default_value = "yaml")]
        format: String,
    },

    /// Calculate MQS score from evidence
    Score {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Model ID for the score
        #[arg(short, long)]
        model: String,
    },

    /// Generate report from execution results
    Report {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Report formats to generate (html, junit, all)
        #[arg(long, default_value = "all")]
        formats: String,

        /// Model ID
        #[arg(short, long)]
        model: String,
    },

    /// List available models in registry
    List {
        /// Filter by size category (small, medium, large, xlarge)
        #[arg(short, long)]
        size: Option<String>,
    },

    /// Lock playbook hashes for integrity verification (§3.1)
    LockPlaybooks {
        /// Directory containing playbook YAML files
        #[arg(value_name = "DIR", default_value = "playbooks")]
        dir: PathBuf,

        /// Output lock file path
        #[arg(short, long, default_value = "playbooks/playbook.lock.yaml")]
        output: PathBuf,
    },

    /// Generate upstream tickets from failures
    Tickets {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Target repository (e.g., "paiml/aprender")
        #[arg(short, long, default_value = "paiml/aprender")]
        repo: String,

        /// Only generate tickets for black swan events
        #[arg(long)]
        black_swans_only: bool,

        /// Minimum occurrences before creating ticket
        #[arg(long, default_value = "1")]
        min_occurrences: usize,

        /// Ticket generation mode (F-TICKET-004)
        /// - create: Generate ticket files (default)
        /// - draft: Only print ticket content without creating files
        #[arg(long, default_value = "create")]
        ticket_mode: String,
    },

    /// Verify model output parity against HuggingFace golden corpus
    ///
    /// Implements Popperian falsification: any divergence beyond tolerance
    /// falsifies the hypothesis that the implementation is equivalent to HuggingFace.
    Parity {
        /// Model family (e.g., "qwen2.5-coder-1.5b")
        #[arg(short, long)]
        model_family: String,

        /// Path to golden corpus directory
        #[arg(short, long, default_value = "../hf-ground-truth-corpus/oracle")]
        corpus_path: PathBuf,

        /// SafeTensors file containing logits to verify
        #[arg(short, long)]
        logits_file: Option<PathBuf>,

        /// Prompt used to generate the logits
        #[arg(short, long)]
        prompt: Option<String>,

        /// Tolerance level (fp32, fp16, int8, int4)
        #[arg(short, long, default_value = "fp32")]
        tolerance: String,

        /// List available golden outputs for the model
        #[arg(long)]
        list: bool,

        /// Verify all golden outputs against themselves (sanity check)
        #[arg(long)]
        self_check: bool,
    },

    /// Export certification data to models.csv (PMAT-264)
    ///
    /// Scans evidence directory and updates models.csv with MQS scores,
    /// grades, and certification status for oracle consumption.
    ExportCsv {
        /// Directory containing evidence JSON files
        #[arg(short, long, default_value = "docs/certifications/evidence")]
        evidence_dir: PathBuf,

        /// Output CSV file path
        #[arg(short, long, default_value = "docs/certifications/models.csv")]
        output: PathBuf,

        /// Append to existing CSV (instead of overwrite)
        #[arg(long)]
        append: bool,
    },

    /// Export evidence to schema-compliant JSON (PMAT-265)
    ///
    /// Exports test run results to the standard evidence JSON format
    /// consumed by the oracle for certification lookup.
    ExportEvidence {
        /// Path to source evidence or execution result JSON
        #[arg(value_name = "SOURCE")]
        source: PathBuf,

        /// Output directory for evidence files
        #[arg(short, long, default_value = "docs/certifications/evidence")]
        output_dir: PathBuf,

        /// Model HF repo ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        #[arg(short, long)]
        model: String,

        /// Model family (e.g., "qwen2")
        #[arg(long)]
        family: String,

        /// Model size (e.g., "0.5b")
        #[arg(long)]
        size: String,

        /// Playbook name
        #[arg(long)]
        playbook_name: String,

        /// Certification tier (smoke, mvp, full)
        #[arg(long, default_value = "mvp")]
        tier: String,
    },
}

/// Setup SIGINT handler for Jidoka cleanup
///
/// Toyota Way: Stop the line, clean up, never leave orphan processes.
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = apr_qa_runner::process::kill_all_registered();
        eprintln!("\n[JIDOKA] SIGINT received. Reaping {count} child process(es)...");
        eprintln!("[JIDOKA] Toyota Way: Stop the line, clean up, exit.");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Failed to set signal handler: {e}");
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    setup_signal_handler();

    let cli = Cli::parse();

    match cli.command {
        Commands::Certify {
            all,
            family,
            tier,
            models,
            output,
            dry_run,
            model_cache,
            apr_binary,
            auto_ticket,
            ticket_repo,
            no_integrity_check,
            fail_fast,
            oracle_enhance,
        } => {
            run_certification(
                all,
                family,
                &tier,
                &models,
                &output,
                dry_run,
                model_cache,
                &apr_binary,
                auto_ticket,
                &ticket_repo,
                no_integrity_check,
                fail_fast,
                oracle_enhance,
            );
        }
        Commands::Run {
            playbook,
            output,
            failure_policy,
            fail_fast,
            dry_run,
            workers,
            model_path,
            timeout,
            no_gpu,
            skip_conversion_tests,
            run_tool_tests,
            profile_ci,
            no_differential,
            no_trace_payload,
            hf_parity,
            hf_corpus_path,
            hf_model_family,
            no_integrity_check,
        } => {
            // --fail-fast flag overrides --failure-policy
            let effective_policy = if fail_fast {
                "fail-fast".to_string()
            } else {
                failure_policy
            };
            run_playbook(
                &playbook,
                &output,
                &effective_policy,
                dry_run,
                workers,
                model_path,
                timeout,
                no_gpu,
                skip_conversion_tests,
                run_tool_tests,
                profile_ci,
                no_differential,
                no_trace_payload,
                hf_parity,
                &hf_corpus_path,
                hf_model_family,
                no_integrity_check,
            );
        }
        Commands::Tools {
            model_path,
            no_gpu,
            output,
            include_serve,
        } => {
            run_tool_tests(&model_path, no_gpu, &output, include_serve);
        }
        Commands::Generate {
            model,
            count,
            format,
        } => {
            generate_scenarios(&model, count, &format);
        }
        Commands::Score { evidence, model } => {
            calculate_score(&evidence, &model);
        }
        Commands::Report {
            evidence,
            output,
            formats,
            model,
        } => {
            generate_report(&evidence, &output, &formats, &model);
        }
        Commands::List { size } => {
            list_models(size.as_deref());
        }
        Commands::LockPlaybooks { dir, output } => match generate_lock_file(&dir, &output) {
            Ok(count) => println!("Locked {count} playbook(s) → {}", output.display()),
            Err(e) => {
                eprintln!("Error generating lock file: {e}");
                std::process::exit(1);
            }
        },
        Commands::Tickets {
            evidence,
            repo,
            black_swans_only,
            min_occurrences,
            ticket_mode,
        } => {
            generate_tickets(
                &evidence,
                &repo,
                black_swans_only,
                min_occurrences,
                &ticket_mode,
            );
        }
        Commands::Parity {
            model_family,
            corpus_path,
            logits_file,
            prompt,
            tolerance,
            list,
            self_check,
        } => {
            run_parity_check(
                &model_family,
                &corpus_path,
                logits_file.as_deref(),
                prompt.as_deref(),
                &tolerance,
                list,
                self_check,
            );
        }
        Commands::ExportCsv {
            evidence_dir,
            output,
            append,
        } => {
            export_csv(&evidence_dir, &output, append);
        }
        Commands::ExportEvidence {
            source,
            output_dir,
            model,
            family,
            size,
            playbook_name,
            tier,
        } => {
            export_evidence(
                &source,
                &output_dir,
                &model,
                &family,
                &size,
                &playbook_name,
                &tier,
            );
        }
    }
}

#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn run_playbook(
    playbook_path: &PathBuf,
    output_dir: &PathBuf,
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests_flag: bool,
    profile_ci: bool,
    no_differential: bool,
    no_trace_payload: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<String>,
    no_integrity_check: bool,
) {
    // Log environment for fail-fast mode (§12.5.3)
    if failure_policy == "fail-fast" {
        log_environment();
    }

    println!("Loading playbook: {}", playbook_path.display());

    let playbook = match load_playbook(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if !no_integrity_check {
        verify_playbook_lock_or_exit(playbook_path, &playbook.name);
    }

    // Model path resolution is handled by G0-PULL in the executor.
    // G0-PULL calls `apr pull --json <hf_repo>` which is the authoritative
    // source for model location (Five-Whys: GH-190). The previous HF-CACHE-001
    // auto-resolution via resolve_hf_repo_to_cache() could pick up corrupt
    // cached models (e.g., 2.4GB F32 zeros in HF cache).

    // Validate failure policy
    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }

    // §3.4: Resource-aware scheduling - enforce worker limits based on model size
    let effective_workers = playbook.effective_max_workers(workers);
    if effective_workers < workers {
        eprintln!(
            "[RESOURCE] Model size {:?} caps workers at {} (requested {})",
            playbook.size_category(),
            effective_workers,
            workers
        );
    }

    println!("Running playbook: {}", playbook.name);
    println!("  Total tests: {}", playbook.total_tests());
    println!("  Dry run: {dry_run}");
    println!("  Model size: {:?}", playbook.size_category());
    if let Some(ref path) = model_path {
        println!("  Model path: {path}");
    }
    println!(
        "  Workers: {} (max for size: {})",
        effective_workers,
        playbook.model.size_category.max_workers()
    );
    println!("  Timeout: {timeout}ms");

    let run_config = PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers: effective_workers, // §3.4: Use enforced worker limit
        model_path: model_path.clone(),
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests: run_tool_tests_flag,
        run_differential_tests: !no_differential,
        run_profile_ci: profile_ci,
        run_trace_payload: !no_trace_payload,
        run_hf_parity: hf_parity,
        hf_parity_corpus_path: if hf_parity {
            Some(hf_corpus_path.to_string())
        } else {
            None
        },
        hf_parity_model_family: hf_model_family.clone(),
    };

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Print conversion test status (P0 CRITICAL)
    if !skip_conversion_tests && model_path.is_some() {
        println!("  Conversion tests: ENABLED (P0 CRITICAL)");
    } else if skip_conversion_tests {
        println!("  Conversion tests: DISABLED (WARNING: P0 tests skipped)");
    }

    // Print HF parity status
    if hf_parity {
        println!("  HF parity: ENABLED");
        println!("    Corpus: {hf_corpus_path}");
        if let Some(ref family) = hf_model_family {
            println!("    Model family: {family}");
        } else {
            println!("    Model family: NOT SET (required for parity tests)");
        }
    }

    // Run tool tests if enabled
    if run_tool_tests_flag {
        if let Some(ref mp) = model_path {
            println!("\n=== Running APR Tool Tests ===");
            let tool_executor = ToolExecutor::new(mp.clone(), no_gpu, timeout);
            let tool_results = tool_executor.execute_all();
            let tool_passed = tool_results.iter().filter(|r| r.passed).count();
            let tool_failed = tool_results.len() - tool_passed;
            println!("  Tool tests: {tool_passed} passed, {tool_failed} failed");
        }
    }

    let result = match execute_playbook(&playbook, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    print_playbook_results(&result);
    save_playbook_evidence(&result, output_dir);
}

fn print_playbook_results(result: &apr_qa_runner::ExecutionResult) {
    println!("\n=== Execution Results ===");
    println!("  Total scenarios: {}", result.total_scenarios);
    println!("  Passed: {}", result.passed);
    println!("  Failed: {}", result.failed);
    println!("  Skipped: {}", result.skipped);
    println!("  Duration: {}ms", result.duration_ms);
    println!("  Pass rate: {:.1}%", result.pass_rate());

    if let Some(ref gateway_fail) = result.gateway_failed {
        println!("  Gateway FAILED: {gateway_fail}");
    }
}

fn save_playbook_evidence(result: &apr_qa_runner::ExecutionResult, output_dir: &PathBuf) {
    // GH-212: If --output ends with .json, treat as file path, not directory
    let evidence_path = if output_dir
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Error creating output directory: {e}");
            return;
        }
        output_dir.clone()
    } else {
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            eprintln!("Error creating output directory: {e}");
            return;
        }
        output_dir.join("evidence.json")
    };
    match result.evidence.to_json() {
        Ok(json) => {
            if let Err(e) = std::fs::write(&evidence_path, json) {
                eprintln!("Error writing evidence: {e}");
            } else {
                println!("\nEvidence saved to: {}", evidence_path.display());
            }
        }
        Err(e) => eprintln!("Error serializing evidence: {e}"),
    }
}

/// Log environment information for fail-fast diagnostics (§12.5.3)
fn log_environment() {
    eprintln!("\n[ENVIRONMENT] === Diagnostic Context ===");
    eprintln!(
        "[ENVIRONMENT] OS: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    eprintln!(
        "[ENVIRONMENT] apr-qa version: {}",
        env!("CARGO_PKG_VERSION")
    );

    // Git context
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let commit = String::from_utf8_lossy(&output.stdout);
            eprintln!("[ENVIRONMENT] Git commit: {}", commit.trim());
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout);
            eprintln!("[ENVIRONMENT] Git branch: {}", branch.trim());
        }
    }

    // Check for dirty files
    if let Ok(output) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
    {
        if output.status.success() {
            let status = String::from_utf8_lossy(&output.stdout);
            let dirty_count = status.lines().count();
            if dirty_count > 0 {
                eprintln!("[ENVIRONMENT] Git dirty: {dirty_count} file(s) modified");
            }
        }
    }

    // apr CLI version
    if let Ok(output) = std::process::Command::new("apr").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("[ENVIRONMENT] apr-cli: {}", version.trim());
        }
    }

    // Rust version
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("[ENVIRONMENT] {}", version.trim());
        }
    }

    eprintln!("[ENVIRONMENT] ===========================\n");
}

fn generate_scenarios(model_id: &str, count: usize, format: &str) {
    let scenarios = generate_model_scenarios(model_id, count);

    println!("Generated {} scenarios for {model_id}", scenarios.len());

    match format {
        "yaml" => match scenarios_to_yaml(&scenarios) {
            Ok(yaml) => println!("{yaml}"),
            Err(e) => eprintln!("{e}"),
        },
        "json" => match scenarios_to_json(&scenarios) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("{e}"),
        },
        _ => {
            eprintln!("Unknown format: {format}");
            std::process::exit(1);
        }
    }
}

fn calculate_score(evidence_path: &PathBuf, model_id: &str) {
    let evidence_json = match std::fs::read_to_string(evidence_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading evidence file: {e}");
            std::process::exit(1);
        }
    };

    let evidence = match parse_evidence(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let collector = collect_evidence(evidence);

    match calculate_mqs_score(model_id, &collector) {
        Ok(score) => {
            println!("=== Model Qualification Score (MQS) ===");
            println!("Model: {}", score.model_id);
            println!("Raw Score: {}/1000", score.raw_score);
            println!("Normalized Score: {:.1}/100", score.normalized_score);
            println!("Grade: {}", score.grade);
            println!("Gateways Passed: {}", score.gateways_passed);
            println!("Qualifies: {}", score.qualifies());
            println!("Production Ready: {}", score.is_production_ready());

            println!("\n--- Category Breakdown ---");
            let breakdown = score.categories.breakdown();
            for (cat, (pts, max)) in &breakdown {
                println!("  {cat}: {pts}/{max}");
            }

            if !score.penalties.is_empty() {
                println!("\n--- Penalties ---");
                for penalty in &score.penalties {
                    println!(
                        "  {}: {} (-{} pts)",
                        penalty.code, penalty.description, penalty.points
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn generate_report(evidence_path: &PathBuf, output_dir: &PathBuf, formats: &str, model_id: &str) {
    let evidence_json = read_file_or_exit(evidence_path, "evidence file");
    let evidence = parse_evidence_or_exit(&evidence_json);
    let collector = collect_evidence(evidence);
    let mqs_score = calculate_mqs_or_exit(model_id, &collector);
    let popperian_score = calculate_popperian_score(model_id, &collector);

    create_dir_or_exit(output_dir);
    write_report_formats(
        output_dir,
        formats,
        model_id,
        &mqs_score,
        &popperian_score,
        &collector,
    );
}

fn read_file_or_exit(path: &PathBuf, desc: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {desc}: {e}");
        std::process::exit(1);
    })
}

fn parse_evidence_or_exit(json: &str) -> Vec<Evidence> {
    parse_evidence(json).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

fn calculate_mqs_or_exit(model_id: &str, collector: &EvidenceCollector) -> MqsScore {
    calculate_mqs_score(model_id, collector).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

fn create_dir_or_exit(dir: &PathBuf) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }
}

fn write_report_formats(
    output_dir: &PathBuf,
    formats: &str,
    model_id: &str,
    mqs_score: &MqsScore,
    popperian_score: &PopperianScore,
    collector: &EvidenceCollector,
) {
    let gen_html = formats == "all" || formats == "html";
    let gen_junit = formats == "all" || formats == "junit";

    if gen_html {
        write_html_report(output_dir, model_id, mqs_score, popperian_score, collector);
    }
    if gen_junit {
        write_junit_report(output_dir, model_id, collector, mqs_score);
    }
    write_mqs_json(output_dir, mqs_score);
}

fn write_html_report(
    output_dir: &PathBuf,
    model_id: &str,
    mqs_score: &MqsScore,
    popperian_score: &PopperianScore,
    collector: &EvidenceCollector,
) {
    let result = generate_html_report(
        &format!("MQS Report: {model_id}"),
        mqs_score,
        popperian_score,
        collector,
    );
    write_report_file(output_dir, "report.html", "HTML report", result);
}

fn write_junit_report(
    output_dir: &PathBuf,
    model_id: &str,
    collector: &EvidenceCollector,
    mqs_score: &MqsScore,
) {
    let result = generate_junit_report(model_id, collector, mqs_score);
    write_report_file(output_dir, "junit.xml", "JUnit report", result);
}

fn write_report_file<E: std::fmt::Display>(
    output_dir: &PathBuf,
    filename: &str,
    desc: &str,
    result: Result<String, E>,
) {
    match result {
        Ok(content) => {
            let path = output_dir.join(filename);
            match std::fs::write(&path, content) {
                Ok(()) => println!("{desc}: {}", path.display()),
                Err(e) => eprintln!("Error writing {desc}: {e}"),
            }
        }
        Err(e) => eprintln!("{e}"),
    }
}

fn write_mqs_json(output_dir: &PathBuf, mqs_score: &MqsScore) {
    let score_path = output_dir.join("mqs.json");
    match serde_json::to_string_pretty(mqs_score) {
        Ok(json) => match std::fs::write(&score_path, json) {
            Ok(()) => println!("MQS score: {}", score_path.display()),
            Err(e) => eprintln!("Error writing MQS JSON: {e}"),
        },
        Err(e) => eprintln!("Error serializing MQS: {e}"),
    }
}

fn list_models(size_filter: Option<&str>) {
    let models = list_all_models();

    println!("=== Available Models ===\n");

    let filtered_models = if let Some(filter) = size_filter {
        filter_models_by_size(&models, filter)
    } else {
        models
    };

    for model in filtered_models {
        println!("  {} ({:?})", model.id.hf_repo(), model.size);
    }
}

fn generate_tickets(
    evidence_path: &PathBuf,
    repo: &str,
    black_swans_only: bool,
    min_occurrences: usize,
    ticket_mode: &str,
) {
    let is_draft = ticket_mode == "draft";

    let evidence_json = match std::fs::read_to_string(evidence_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading evidence file: {e}");
            std::process::exit(1);
        }
    };

    let evidence = match parse_evidence(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let tickets =
        generate_tickets_from_evidence(&evidence, repo, black_swans_only, min_occurrences);

    if is_draft {
        // F-TICKET-004: Draft mode - only print, don't create files
        println!("=== Ticket Drafts ({}) ===", tickets.len());
        println!("(Draft mode: No files created)\n");

        for ticket in &tickets {
            println!("--- {} ---", ticket.title);
            println!("Priority: {}", ticket.priority);
            println!("Category: {}", ticket.category);
            println!("Labels: {}", ticket.labels.join(", "));
            println!();
            println!("Body:");
            println!("{}", ticket.body);
            println!();
            println!("gh command (would run):");
            println!("  {}\n", ticket.to_gh_command(repo));
            println!("{}", "=".repeat(60));
        }
    } else {
        // Create mode - generate files and show commands
        println!("=== Generated Tickets ({}) ===\n", tickets.len());

        for ticket in &tickets {
            println!("--- {} ---", ticket.title);
            println!("Priority: {}", ticket.priority);
            println!("Category: {}", ticket.category);
            println!("Labels: {}", ticket.labels.join(", "));
            println!();
            println!("gh command:");
            println!("  {}\n", ticket.to_gh_command(repo));
        }
    }
}

/// Run HF Parity Oracle verification
///
/// Implements Popperian falsification: any divergence beyond tolerance
/// falsifies the hypothesis that the implementation is equivalent to HuggingFace.
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
fn run_parity_check(
    model_family: &str,
    corpus_path: &std::path::Path,
    logits_file: Option<&std::path::Path>,
    prompt: Option<&str>,
    tolerance_str: &str,
    list: bool,
    self_check: bool,
) {
    use apr_qa_gen::{HfParityOracle, Tolerance};

    println!("=== HuggingFace Parity Oracle ===\n");
    println!("Model family: {model_family}");
    println!("Corpus path: {}", corpus_path.display());

    // Parse tolerance
    let tolerance = match tolerance_str.to_lowercase().as_str() {
        "fp32" => Tolerance::fp32(),
        "fp16" => Tolerance::fp16(),
        "int8" => Tolerance::int8(),
        "int4" => Tolerance::int4(),
        _ => {
            eprintln!("Unknown tolerance level: {tolerance_str}");
            eprintln!("Valid options: fp32, fp16, int8, int4");
            std::process::exit(1);
        }
    };
    println!("Tolerance: {tolerance_str}");

    // Create oracle
    let oracle = HfParityOracle::new(corpus_path, model_family).with_tolerance(tolerance);

    // Check corpus exists
    let corpus_dir = corpus_path.join(model_family);
    if !corpus_dir.exists() {
        eprintln!(
            "\nError: Corpus directory not found: {}",
            corpus_dir.display()
        );
        eprintln!("Available models:");
        if let Ok(entries) = std::fs::read_dir(corpus_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    println!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        std::process::exit(1);
    }

    if list {
        parity_list_golden(&corpus_dir);
        return;
    }

    if self_check {
        parity_self_check(&oracle, &corpus_dir);
        return;
    }

    parity_verify(&oracle, logits_file, prompt, tolerance_str);
}

/// List available golden outputs in the corpus directory
fn parity_list_golden(corpus_dir: &std::path::Path) {
    println!("\n=== Available Golden Outputs ===\n");
    let manifest_path = corpus_dir.join("manifest.json");
    if !manifest_path.exists() {
        return;
    }
    let Ok(content) = std::fs::read_to_string(&manifest_path) else {
        return;
    };
    let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) else {
        return;
    };
    let Some(prompts) = manifest.get("prompts").and_then(|p| p.as_array()) else {
        return;
    };
    println!("Found {} golden outputs:\n", prompts.len());

    for entry in std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json")
            || path.file_stem().is_none_or(|s| s == "manifest")
        {
            continue;
        }
        if let Some((hash, prompt_str)) = read_golden_prompt(&path) {
            let truncated = truncate_str(&prompt_str, 50);
            println!("  [{hash}] {truncated}");
        }
    }
}

/// Read the prompt from a golden output JSON file, returning (hash, prompt)
fn read_golden_prompt(path: &std::path::Path) -> Option<(String, String)> {
    let json = std::fs::read_to_string(path).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&json).ok()?;
    let prompt = meta.get("prompt")?.as_str()?.to_string();
    let hash = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Some((hash, prompt))
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Self-check mode: verify golden outputs match themselves
fn parity_self_check(oracle: &apr_qa_gen::HfParityOracle, corpus_dir: &std::path::Path) {
    println!("\n=== Self-Check Mode ===");
    println!("Verifying golden outputs match themselves (sanity check)...\n");

    let mut passed = 0;
    let mut failed = 0;

    for entry in std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json")
            || path.file_stem().is_none_or(|s| s == "manifest")
        {
            continue;
        }
        let Some((_, prompt_str)) = read_golden_prompt(&path) else {
            continue;
        };
        match oracle.load_golden(&prompt_str) {
            Ok(golden) => match oracle.tensors_close(&golden.logits, &golden.logits) {
                Ok(()) => {
                    passed += 1;
                    println!("  ✓ {}", truncate_str(&prompt_str, 40));
                }
                Err(diff) => {
                    failed += 1;
                    eprintln!("  ✗ {prompt_str}: {diff}");
                }
            },
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ Failed to load {prompt_str}: {e}");
            }
        }
    }

    println!("\n=== Self-Check Results ===");
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Verification mode: compare a logits file against golden reference
fn parity_verify(
    oracle: &apr_qa_gen::HfParityOracle,
    logits_file: Option<&std::path::Path>,
    prompt: Option<&str>,
    tolerance_str: &str,
) {
    use apr_qa_gen::hash_prompt;

    let Some(logits_path) = logits_file else {
        eprintln!("\nError: --logits-file is required for verification");
        eprintln!("Use --list to see available golden outputs");
        eprintln!("Use --self-check to verify corpus integrity");
        std::process::exit(1);
    };

    let Some(prompt_str) = prompt else {
        eprintln!("\nError: --prompt is required for verification");
        std::process::exit(1);
    };

    println!("\n=== Verification Mode ===");
    println!("Prompt: {prompt_str}");
    println!("Logits file: {}", logits_path.display());

    let logits = load_logits_from_file(logits_path);

    match oracle.load_golden(prompt_str) {
        Ok(golden) => {
            println!("\nGolden output found:");
            println!("  Model: {}", golden.model_id);
            println!("  Transformers version: {}", golden.transformers_version);
            println!("  Shape: {:?}", golden.shape);
            println!("  Input hash: {}", hash_prompt(prompt_str));
            println!(
                "\nComparing logits ({} vs {} elements)...",
                logits.len(),
                golden.logits.len()
            );

            match oracle.tensors_close(&logits, &golden.logits) {
                Ok(()) => {
                    println!("\n✓ PARITY VERIFIED");
                    println!("  Logits are within tolerance ({tolerance_str})");
                    println!("  Hypothesis corroborated: implementation matches HuggingFace");
                }
                Err(diff) => {
                    eprintln!("\n✗ PARITY FALSIFIED");
                    eprintln!("  {diff}");
                    eprintln!("\n  Interpretation (Popper, 1959):");
                    eprintln!("  The hypothesis that this implementation produces");
                    eprintln!("  equivalent outputs to HuggingFace has been falsified.");
                    eprintln!("  Investigation required before certification can proceed.");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Error loading golden output: {e}");
            eprintln!("\nHint: Use --list to see available golden outputs");
            std::process::exit(1);
        }
    }
}

/// Load logits tensor from a SafeTensors file
fn load_logits_from_file(logits_path: &std::path::Path) -> Vec<f32> {
    let logits_data = match std::fs::read(logits_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading logits file: {e}");
            std::process::exit(1);
        }
    };

    let tensors = match safetensors::SafeTensors::deserialize(&logits_data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error parsing SafeTensors: {e}");
            std::process::exit(1);
        }
    };

    let logits_view = match tensors.tensor("logits") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: 'logits' tensor not found: {e}");
            std::process::exit(1);
        }
    };

    logits_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn run_tool_tests(
    model_path: &std::path::Path,
    no_gpu: bool,
    output_dir: &std::path::Path,
    include_serve: bool,
) {
    use apr_qa_runner::ToolExecutor;

    println!("=== APR Tool Coverage Tests ===\n");
    println!("Model: {}", model_path.display());
    println!("GPU: {}", if no_gpu { "disabled" } else { "enabled" });
    println!(
        "Serve test: {}\n",
        if include_serve { "enabled" } else { "disabled" }
    );

    let executor = ToolExecutor::new(model_path.to_string_lossy().to_string(), no_gpu, 120_000);

    let results = executor.execute_all_with_serve(include_serve);

    let mut passed = 0;
    let mut failed = 0;

    println!("{:<20} {:<10} {:<10} Duration", "Tool", "Status", "Exit");
    println!("{}", "-".repeat(60));

    for result in &results {
        let status = if result.passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };
        println!(
            "{:<20} {:<10} {:<10} {}ms",
            result.tool, status, result.exit_code, result.duration_ms
        );

        if result.passed {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("{}", "-".repeat(60));
    println!("Total: {passed} passed, {failed} failed\n");

    // Save results to JSON
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        return;
    }

    let results_json = serde_json::to_string_pretty(
        &results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "tool": r.tool,
                    "passed": r.passed,
                    "exit_code": r.exit_code,
                    "duration_ms": r.duration_ms,
                    "gate_id": r.gate_id,
                    "stderr": r.stderr,
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_default();

    let results_path = output_dir.join("tool_tests.json");
    if let Err(e) = std::fs::write(&results_path, results_json) {
        eprintln!("Error saving tool test results: {e}");
    } else {
        println!("Results saved to: {}", results_path.display());
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::fn_params_excessive_bools)]
fn run_certification(
    all: bool,
    family: Option<String>,
    tier_str: &str,
    model_ids: &[String],
    output_dir: &PathBuf,
    dry_run: bool,
    model_cache: Option<PathBuf>,
    apr_binary: &str,
    auto_ticket: bool,
    ticket_repo: &str,
    no_integrity_check: bool,
    fail_fast: bool,
    oracle_enhance: bool,
) {
    use apr_qa_certify::write_csv;

    let tier: CertTier = match tier_str.parse() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let model_cache = resolve_default_model_cache(model_cache);

    if fail_fast {
        log_environment();
    }

    print_certification_header(tier_str, dry_run, fail_fast, model_cache.as_ref());

    let (csv_path, mut certifications) = load_certification_csv();
    let models_to_certify =
        determine_models_to_certify(all, family.as_deref(), model_ids, &certifications);

    println!("Models to certify: {}\n", models_to_certify.len());

    if dry_run {
        for model_id in &models_to_certify {
            let playbook_name = playbook_path_for_model(model_id, tier);
            println!("  Would certify: {model_id}");
            println!("    Playbook: {playbook_name}");
        }
        return;
    }

    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    let (certified_count, failed_count) = certify_model_loop(
        &models_to_certify,
        tier,
        tier_str,
        model_cache.as_ref(),
        apr_binary,
        no_integrity_check,
        fail_fast,
        oracle_enhance,
        output_dir,
        &mut certifications,
    );

    let csv_output = write_csv(&certifications);
    if let Err(e) = std::fs::write(&csv_path, &csv_output) {
        eprintln!("Error writing models.csv: {e}");
    } else {
        println!("Updated: {}", csv_path.display());
    }

    warn_missing_lock_file(no_integrity_check);

    if auto_ticket {
        run_auto_ticket_generation(&models_to_certify, output_dir, ticket_repo);
    }

    println!("\n=== Certification Summary ===");
    println!("Certified: {certified_count}");
    println!("Failed: {failed_count}");
    println!("Total: {}", models_to_certify.len());
}

fn resolve_default_model_cache(model_cache: Option<PathBuf>) -> Option<PathBuf> {
    if model_cache.is_some() {
        return model_cache;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let default_cache = PathBuf::from(home).join(".cache/apr-models");
    println!("Auto-resolving model cache: {}", default_cache.display());
    Some(default_cache)
}

fn print_certification_header(
    tier_str: &str,
    dry_run: bool,
    fail_fast: bool,
    model_cache: Option<&PathBuf>,
) {
    println!("=== APR Model Certification ===\n");
    println!("Tier: {tier_str}");
    println!("Dry run: {dry_run}");
    println!("Fail-fast: {fail_fast}");
    if let Some(cache) = model_cache {
        println!("Model cache: {}", cache.display());
    }
    println!();
}

fn load_certification_csv() -> (PathBuf, Vec<apr_qa_certify::ModelCertification>) {
    use apr_qa_certify::parse_csv;
    let csv_path = PathBuf::from("docs/certifications/models.csv");
    let csv_content = match std::fs::read_to_string(&csv_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading models.csv: {e}");
            std::process::exit(1);
        }
    };
    let certifications = match parse_csv(&csv_content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing models.csv: {e}");
            std::process::exit(1);
        }
    };
    (csv_path, certifications)
}

fn determine_models_to_certify(
    all: bool,
    family: Option<&str>,
    model_ids: &[String],
    certifications: &[apr_qa_certify::ModelCertification],
) -> Vec<String> {
    if all {
        certifications.iter().map(|c| c.model_id.clone()).collect()
    } else if let Some(fam) = family {
        certifications
            .iter()
            .filter(|c| c.family == fam)
            .map(|c| c.model_id.clone())
            .collect()
    } else if !model_ids.is_empty() {
        model_ids.to_vec()
    } else {
        eprintln!("Error: Specify --all, --family, or model IDs");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn certify_model_loop(
    models_to_certify: &[String],
    tier: CertTier,
    tier_str: &str,
    model_cache: Option<&PathBuf>,
    apr_binary: &str,
    no_integrity_check: bool,
    fail_fast: bool,
    oracle_enhance: bool,
    output_dir: &PathBuf,
    certifications: &mut [apr_qa_certify::ModelCertification],
) -> (usize, usize) {
    let mut certified_count = 0;
    let mut failed_count = 0;

    for model_id in models_to_certify {
        let short: &str = model_id.split('/').next_back().unwrap_or(model_id);
        let playbook_name = playbook_path_for_model(model_id, tier);

        println!("--- Certifying: {model_id} ---");
        println!("  Playbook: {playbook_name}");

        if let Some(cache) = model_cache {
            let model_dir = cache.join(short.to_lowercase().replace('.', "-"));
            auto_populate_model_cache(model_id, &model_dir, apr_binary);
        }

        let playbook_path = std::path::Path::new(&playbook_name);
        if !playbook_path.exists() {
            eprintln!("  Playbook not found, skipping");
            failed_count += 1;
            continue;
        }

        let playbook = match load_playbook(playbook_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Error loading playbook: {e}");
                failed_count += 1;
                continue;
            }
        };

        if !verify_playbook_lock(playbook_path, &playbook.name, no_integrity_check) {
            failed_count += 1;
            continue;
        }

        let model_cache_path = model_cache.map(|cache| {
            cache
                .join(short.to_lowercase().replace('.', "-"))
                .to_string_lossy()
                .to_string()
        });

        let config = build_certification_config_with_policy(tier, model_cache_path, fail_fast);
        let should_break = process_certification_result(
            model_id,
            &playbook,
            config,
            tier,
            tier_str,
            model_cache,
            apr_binary,
            fail_fast,
            oracle_enhance,
            output_dir,
            certifications,
            short,
            &mut certified_count,
            &mut failed_count,
        );

        if should_break {
            break;
        }
    }

    (certified_count, failed_count)
}

/// Verify playbook lock or exit (for `run` subcommand)
fn verify_playbook_lock_or_exit(playbook_path: &std::path::Path, playbook_name: &str) {
    let lock_path = std::path::Path::new("playbooks/playbook.lock.yaml");
    if lock_path.exists() {
        match apr_qa_runner::load_lock_file(lock_path) {
            Ok(lock_file) => {
                if let Err(e) = apr_qa_runner::verify_playbook_integrity(
                    playbook_path,
                    &lock_file,
                    playbook_name,
                ) {
                    eprintln!("[INTEGRITY] {e}");
                    eprintln!("[INTEGRITY] Playbook hash does not match lock file.");
                    eprintln!("[INTEGRITY] Either:");
                    eprintln!("  1. Run `apr-qa lock-playbooks` to regenerate the lock file");
                    eprintln!("  2. Use --no-integrity-check to bypass (NOT RECOMMENDED)");
                    std::process::exit(1);
                }
                println!("  Integrity check: PASSED");
            }
            Err(e) => {
                eprintln!("[WARN] Could not load lock file: {e}");
            }
        }
    } else {
        eprintln!(
            "[WARN] No playbook lock file found. Run `apr-qa lock-playbooks` to generate one."
        );
    }
}

/// Returns true if playbook integrity is verified (or check skipped), false if blocked
fn verify_playbook_lock(
    playbook_path: &std::path::Path,
    playbook_name: &str,
    no_integrity_check: bool,
) -> bool {
    if no_integrity_check {
        return true;
    }
    let lock_path = std::path::Path::new("playbooks/playbook.lock.yaml");
    if !lock_path.exists() {
        return true;
    }
    match apr_qa_runner::load_lock_file(lock_path) {
        Ok(lock_file) => {
            if let Err(e) =
                apr_qa_runner::verify_playbook_integrity(playbook_path, &lock_file, playbook_name)
            {
                eprintln!("  [INTEGRITY] {e}");
                eprintln!(
                    "  [INTEGRITY] CERTIFICATION BLOCKED: Playbook modified without updating lock file."
                );
                eprintln!(
                    "  [INTEGRITY] Run `apr-qa lock-playbooks` first or use --no-integrity-check"
                );
                return false;
            }
            println!("  Integrity check: PASSED");
            true
        }
        Err(e) => {
            eprintln!("  [WARN] Could not load lock file: {e}");
            true
        }
    }
}

/// Process a single model's certification result. Returns true if the loop should break.
#[allow(clippy::too_many_arguments)]
fn process_certification_result(
    model_id: &str,
    playbook: &apr_qa_runner::Playbook,
    config: apr_qa_runner::ExecutionConfig,
    tier: CertTier,
    tier_str: &str,
    model_cache: Option<&PathBuf>,
    apr_binary: &str,
    fail_fast: bool,
    oracle_enhance: bool,
    output_dir: &PathBuf,
    certifications: &mut [apr_qa_certify::ModelCertification],
    short: &str,
    certified_count: &mut usize,
    failed_count: &mut usize,
) -> bool {
    match execute_playbook(playbook, config) {
        Ok(result) => {
            print_execution_summary(&result);

            let Some((raw_score, status, grade, mqs)) =
                compute_certification_scores(model_id, &result, tier)
            else {
                *failed_count += 1;
                return false;
            };

            println!("  Tier: {tier_str}");
            println!("  MQS Score: {raw_score}/1000");
            println!("  Grade: {grade}");
            println!("  Status: {status}");

            let profile =
                run_profiling_phase(&result, playbook, model_cache, short, apr_binary, fail_fast);

            update_certification_record(
                certifications,
                model_id,
                raw_score,
                &grade,
                status,
                tier_str,
                &mqs,
                &profile,
            );

            let model_output = output_dir.join(short.to_lowercase().replace('.', "-"));
            save_evidence(&model_output, &result);

            if oracle_enhance && result.failed > 0 {
                run_oracle_enhancement(model_id, &result, &model_output);
            }

            *certified_count += 1;
            println!();

            if fail_fast && (result.failed > 0 || result.gateway_failed.is_some()) {
                eprintln!("[FAIL-FAST] Stopping certification after {model_id} (had failures)");
                return true;
            }
            false
        }
        Err(e) => {
            eprintln!("  Execution failed: {e}");
            *failed_count += 1;
            if fail_fast {
                eprintln!("[FAIL-FAST] Stopping certification after {model_id} (execution error)");
                return true;
            }
            false
        }
    }
}

fn print_execution_summary(result: &apr_qa_runner::ExecutionResult) {
    println!("  Scenarios: {}", result.total_scenarios);
    println!("  Passed: {}", result.passed);
    println!("  Failed: {}", result.failed);
    println!("  Pass rate: {:.1}%", result.pass_rate());
}

fn compute_certification_scores(
    model_id: &str,
    result: &apr_qa_runner::ExecutionResult,
    tier: CertTier,
) -> Option<(
    u32,
    apr_qa_certify::CertificationStatus,
    String,
    apr_qa_report::MqsScore,
)> {
    use apr_qa_certify::{CertificationTier, grade_from_tier, score_from_tier, status_from_tier};

    let evidence_vec: Vec<_> = result.evidence.all().to_vec();
    let collector = collect_evidence(evidence_vec);
    let mqs = match calculate_mqs_score(model_id, &collector) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Error calculating MQS: {e}");
            return None;
        }
    };

    let cert_tier = match tier {
        CertTier::Mvp => CertificationTier::Mvp,
        CertTier::Smoke | CertTier::Quick | CertTier::Standard | CertTier::Deep => {
            CertificationTier::Full
        }
    };

    let pass_rate = result.pass_rate() / 100.0;
    let has_p0 = result.gateway_failed.is_some();
    let raw_score = score_from_tier(cert_tier, pass_rate, has_p0);
    let status = status_from_tier(cert_tier, pass_rate, has_p0);
    let grade = grade_from_tier(cert_tier, pass_rate, has_p0);

    Some((raw_score, status, grade.to_string(), mqs))
}

fn run_profiling_phase(
    result: &apr_qa_runner::ExecutionResult,
    playbook: &apr_qa_runner::Playbook,
    model_cache: Option<&PathBuf>,
    short: &str,
    apr_binary: &str,
    fail_fast: bool,
) -> apr_qa_runner::SixColumnProfile {
    let has_failures = result.failed > 0 || result.gateway_failed.is_some();
    let mut profile = apr_qa_runner::SixColumnProfile::default();

    if fail_fast && has_failures {
        eprintln!("\n[FAIL-FAST] Skipping profiling - failures detected");
        eprintln!("[FAIL-FAST] Use evidence above for GitHub ticket\n");
        return profile;
    }

    let Some(cache) = model_cache else {
        return profile;
    };

    let model_dir = cache.join(short.to_lowercase().replace('.', "-"));
    if !model_dir.exists() {
        return profile;
    }

    println!("  Running 6-column profiling...");
    match apr_qa_runner::run_six_column_profile(apr_binary, &model_dir, 1, 2) {
        Ok(p) => {
            profile = p;
            print_profiling_results(&profile);
            check_profiling_assertions(&mut profile, playbook);
        }
        Err(e) => {
            eprintln!("  Profiling failed: {e}");
        }
    }

    profile
}

fn print_profiling_results(profile: &apr_qa_runner::SixColumnProfile) {
    for conv in &profile.conversions {
        let status = if conv.cached {
            "cached"
        } else if conv.success {
            "ok"
        } else {
            "FAILED"
        };
        println!(
            "    {} → {}: {} ({}ms)",
            conv.source_format, conv.target_format, status, conv.duration_ms
        );
        if let Some(ref err) = conv.error {
            if let Some(line) = err.lines().last() {
                println!("      {line}");
            }
        }
    }
    println!("    Throughput (tok/s):");
    for (label, tps) in [
        ("GGUF CPU", profile.tps_gguf_cpu),
        ("GGUF GPU", profile.tps_gguf_gpu),
        ("APR CPU ", profile.tps_apr_cpu),
        ("APR GPU ", profile.tps_apr_gpu),
        ("ST CPU  ", profile.tps_st_cpu),
        ("ST GPU  ", profile.tps_st_gpu),
    ] {
        if let Some(tps) = tps {
            println!("      {label}: {tps:.1}");
        }
    }
    println!("    Total profiling time: {}ms", profile.total_duration_ms);
}

fn check_profiling_assertions(
    profile: &mut apr_qa_runner::SixColumnProfile,
    playbook: &apr_qa_runner::Playbook,
) {
    let Some(ref profile_ci) = playbook.profile_ci else {
        return;
    };
    let cpu_threshold = profile_ci
        .assertions
        .min_throughput_cpu
        .or(profile_ci.assertions.min_throughput)
        .unwrap_or(5.0);
    let gpu_threshold = profile_ci
        .assertions
        .min_throughput_gpu
        .or(profile_ci.assertions.min_throughput)
        .unwrap_or(50.0);

    profile.check_assertions(cpu_threshold, gpu_threshold);

    if !profile.failed_assertions.is_empty() {
        println!("    ⚠️  Assertion failures:");
        for fail in &profile.failed_assertions {
            println!(
                "      {} {}: {:.1} tok/s < {:.1} min",
                fail.format.to_uppercase(),
                fail.backend.to_uppercase(),
                fail.actual_tps,
                fail.min_threshold
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_certification_record(
    certifications: &mut [apr_qa_certify::ModelCertification],
    model_id: &str,
    raw_score: u32,
    grade: &str,
    status: apr_qa_certify::CertificationStatus,
    tier_str: &str,
    mqs: &apr_qa_report::MqsScore,
    profile: &apr_qa_runner::SixColumnProfile,
) {
    use apr_qa_certify::CertificationStatus;
    use chrono::Utc;

    let Some(cert) = certifications.iter_mut().find(|c| c.model_id == model_id) else {
        return;
    };

    let (final_status, final_grade, final_tier) = if profile.failed_assertions.is_empty() {
        (status, grade.to_string(), tier_str.to_string())
    } else {
        println!("  ❌ Certification BLOCKED by throughput assertions");
        (
            CertificationStatus::Blocked,
            "-".to_string(),
            "none".to_string(),
        )
    };

    cert.mqs_score = raw_score;
    cert.grade = final_grade;
    cert.status = final_status;
    cert.certified_tier = final_tier;
    cert.last_certified = Some(Utc::now());

    let gw = &mqs.gateways;
    cert.g1 = gw.first().is_some_and(|g| g.passed);
    cert.g2 = gw.get(1).is_some_and(|g| g.passed);
    cert.g3 = gw.get(2).is_some_and(|g| g.passed);
    cert.g4 = gw.get(3).is_some_and(|g| g.passed);

    cert.tps_gguf_cpu = profile.tps_gguf_cpu;
    cert.tps_gguf_gpu = profile.tps_gguf_gpu;
    cert.tps_apr_cpu = profile.tps_apr_cpu;
    cert.tps_apr_gpu = profile.tps_apr_gpu;
    cert.tps_st_cpu = profile.tps_st_cpu;
    cert.tps_st_gpu = profile.tps_st_gpu;
}

fn save_evidence(model_output: &std::path::Path, result: &apr_qa_runner::ExecutionResult) {
    if let Err(e) = std::fs::create_dir_all(model_output) {
        eprintln!("  Error creating model output dir: {e}");
    }
    let evidence_path = model_output.join("evidence.json");
    if let Ok(json) = result.evidence.to_json() {
        let _ = std::fs::write(&evidence_path, json);
        println!("  Evidence: {}", evidence_path.display());
    }
}

fn run_oracle_enhancement(
    model_id: &str,
    result: &apr_qa_runner::ExecutionResult,
    model_output: &std::path::Path,
) {
    use apr_qa_runner::{OracleEnhancer, generate_checklist_markdown};

    let enhancer = OracleEnhancer::new();
    let failed_evidence = result.evidence.failures();

    if failed_evidence.is_empty() {
        return;
    }

    let context = enhancer.enhance_failure(failed_evidence[0]);

    let total = result.passed + result.failed;
    #[allow(clippy::cast_precision_loss)]
    let pass_rate = if total > 0 {
        (result.passed as f64 / total as f64) * 1000.0
    } else {
        0.0
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mqs = pass_rate as u32;
    let grade = if mqs >= 800 {
        "A"
    } else if mqs >= 600 {
        "B"
    } else if mqs >= 400 {
        "C"
    } else {
        "F"
    };

    let checklist_md =
        generate_checklist_markdown(model_id, mqs, grade, total, result.failed, &context);

    let checklist_path = model_output.join("checklist.md");
    if let Err(e) = std::fs::write(&checklist_path, &checklist_md) {
        eprintln!("  Error writing checklist: {e}");
    } else {
        println!("  Checklist: {}", checklist_path.display());
    }

    if context.oracle_available {
        println!(
            "  Oracle: {} hypotheses, {} cross-refs ({}ms)",
            context.hypotheses.len(),
            context.cross_references.len(),
            context.query_latency_ms
        );
    } else {
        println!("  Oracle: unavailable (using static checklist)");
    }
}

fn warn_missing_lock_file(no_integrity_check: bool) {
    if no_integrity_check {
        return;
    }
    let lock_path = "playbooks/playbook.lock.yaml";
    if !std::path::Path::new(lock_path).exists() {
        eprintln!(
            "[WARN] No playbook lock file found at {lock_path}. Run `apr-qa lock-playbooks` to generate one."
        );
    }
}

fn run_auto_ticket_generation(
    models_to_certify: &[String],
    output_dir: &PathBuf,
    ticket_repo: &str,
) {
    let mut all_evidence: Vec<apr_qa_runner::Evidence> = Vec::new();
    for model_id in models_to_certify {
        let short: &str = model_id.split('/').next_back().unwrap_or(model_id);
        let evidence_path = output_dir
            .join(short.to_lowercase().replace('.', "-"))
            .join("evidence.json");
        if let Ok(json) = std::fs::read_to_string(&evidence_path) {
            if let Ok(ev) = parse_evidence(&json) {
                all_evidence.extend(ev);
            }
        }
    }

    if all_evidence.is_empty() {
        return;
    }

    let tickets = execute_auto_tickets(&all_evidence, ticket_repo);
    if tickets.is_empty() {
        println!("\n[AUTO-TICKET] No structured tickets generated (no classified failures).");
    } else {
        println!("\n=== Auto-Generated Tickets ({}) ===", tickets.len());
        for ticket in &tickets {
            println!("  {} [{}]", ticket.title, ticket.priority);
            if let Some(ref fixture) = ticket.upstream_fixture {
                println!("    Fixture: {fixture}");
            }
        }
    }
}

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
