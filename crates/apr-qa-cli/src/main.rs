//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

use apr_qa_cli::{
    CertTier, PlaybookRunConfig, bootstrap_playbook_from_contract,
    build_certification_config_with_policy, build_execution_config, calculate_mqs_score,
    calculate_popperian_score, collect_evidence, execute_auto_tickets, execute_playbook,
    filter_models_by_size, generate_html_report, generate_junit_report, generate_lock_file,
    generate_model_scenarios, generate_tickets_from_evidence, list_all_models, load_playbook,
    parse_evidence, parse_failure_policy, playbook_path_for_model, scenarios_to_json,
    scenarios_to_yaml,
};
use apr_qa_report::{MqsScore, PopperianScore};
use apr_qa_runner::ToolExecutor;
use apr_qa_runner::{Evidence, EvidenceCollector};
use clap::{Parser, Subcommand};
use colored::Colorize;
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

        /// Certification tier (dim-smoke, smoke, quick, standard, deep)
        #[arg(long, default_value = "quick")]
        tier: String,

        /// Kernel equivalence class (A-E) for batch dim-smoke certification
        #[arg(long)]
        kernel_class: Option<String>,

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

    /// Bootstrap an architecture-aware playbook from family contract
    ///
    /// Generates a playbook with architecture-specific prompts that stress-test
    /// the exact kernel operations each model family exercises (GQA, RoPE, etc.)
    Bootstrap {
        /// Model family name (e.g., "qwen2", "llama", "falcon")
        #[arg(value_name = "FAMILY")]
        family: String,

        /// Model size variant (e.g., "1.5b", "7b", "0.5b")
        #[arg(value_name = "SIZE")]
        size: String,

        /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        #[arg(long)]
        hf_repo: String,

        /// Certification tier
        #[arg(long, default_value = "mvp")]
        tier: String,

        /// Output path for generated playbook YAML
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Path to family contracts directory
        #[arg(long, default_value = "../aprender/contracts/model-families")]
        contracts_path: PathBuf,

        /// Dry run: print YAML to stdout instead of writing to file
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate model against tensor layout contract (Issue #4)
    ///
    /// Checks that an APR model file conforms to the tensor layout contract
    /// from aprender (tensor-layout-v1.yaml). This prevents GH-202 style bugs
    /// where wrong tensor shapes cause garbage output.
    ValidateContract {
        /// Path to APR model file to validate
        #[arg(value_name = "MODEL")]
        model_path: PathBuf,

        /// Path to tensor layout contract YAML
        /// Defaults to ../aprender/contracts/tensor-layout-v1.yaml
        #[arg(long)]
        contract_path: Option<PathBuf>,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Only check critical tensors (lm_head, etc.)
        #[arg(long)]
        critical_only: bool,
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
            kernel_class,
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
                kernel_class,
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
        Commands::Bootstrap {
            family,
            size,
            hf_repo,
            tier,
            output,
            contracts_path,
            dry_run,
        } => {
            run_bootstrap(
                &family,
                &size,
                &hf_repo,
                &tier,
                output.as_deref(),
                &contracts_path,
                dry_run,
            );
        }
        Commands::ValidateContract {
            model_path,
            contract_path,
            format,
            critical_only,
        } => {
            validate_contract_command(
                &model_path,
                contract_path.as_deref(),
                &format,
                critical_only,
            );
        }
    }
}

#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn print_run_status(
    playbook: &apr_qa_runner::Playbook,
    effective_workers: usize,
    model_path: Option<&str>,
    dry_run: bool,
    timeout: u64,
    skip_conversion_tests: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<&str>,
) {
    println!(
        "{} {}",
        "Running playbook:".bold(),
        playbook.name.bold().cyan()
    );
    println!("  {} {}", "Total tests:".dimmed(), playbook.total_tests());
    println!("  {} {dry_run}", "Dry run:".dimmed());
    println!(
        "  {} {:?}",
        "Model size:".dimmed(),
        playbook.size_category()
    );
    if let Some(path) = model_path {
        println!("  {} {path}", "Model path:".dimmed());
    }
    println!(
        "  {} {} (max for size: {})",
        "Workers:".dimmed(),
        effective_workers,
        playbook.model.size_category.max_workers()
    );
    println!("  {} {timeout}ms", "Timeout:".dimmed());

    // Conversion test status (P0 CRITICAL)
    if !skip_conversion_tests && model_path.is_some() {
        println!(
            "  {} {}",
            "Conversion tests:".dimmed(),
            "ENABLED (P0 CRITICAL)".bold().green()
        );
    } else if skip_conversion_tests {
        println!(
            "  {} {}",
            "Conversion tests:".dimmed(),
            "DISABLED (WARNING: P0 tests skipped)".bold().yellow()
        );
    }

    // HF parity status
    if hf_parity {
        println!("  {} {}", "HF parity:".dimmed(), "ENABLED".green());
        println!("    {} {hf_corpus_path}", "Corpus:".dimmed());
        if let Some(family) = hf_model_family {
            println!("    {} {family}", "Model family:".dimmed());
        } else {
            println!(
                "    {} {}",
                "Model family:".dimmed(),
                "NOT SET (required for parity tests)".yellow()
            );
        }
    }
}

#[allow(clippy::fn_params_excessive_bools)]
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
    if failure_policy == "fail-fast" {
        log_environment();
    }

    println!(
        "{} {}",
        "Loading playbook:".bold().cyan(),
        playbook_path.display().to_string().cyan()
    );

    let playbook = match load_playbook(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e.red());
            std::process::exit(1);
        }
    };

    if !no_integrity_check {
        verify_playbook_lock_or_exit(playbook_path, &playbook.name);
    }

    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }

    // §3.4: Resource-aware scheduling - enforce worker limits based on model size
    let effective_workers = playbook.effective_max_workers(workers);
    if effective_workers < workers {
        eprintln!(
            "{} Model size {:?} caps workers at {} (requested {})",
            "[RESOURCE]".yellow(),
            playbook.size_category(),
            effective_workers,
            workers
        );
    }

    print_run_status(
        &playbook,
        effective_workers,
        model_path.as_deref(),
        dry_run,
        timeout,
        skip_conversion_tests,
        hf_parity,
        hf_corpus_path,
        hf_model_family.as_deref(),
    );

    let run_config = PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers: effective_workers,
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
        hf_parity_model_family: hf_model_family,
    };

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Run tool tests if enabled
    if run_tool_tests_flag {
        if let Some(ref mp) = model_path {
            println!("\n{}", "=== Running APR Tool Tests ===".bold().cyan());
            let tool_executor = ToolExecutor::new(mp.clone(), no_gpu, timeout);
            let tool_results = tool_executor.execute_all();
            let tool_passed = tool_results.iter().filter(|r| r.passed).count();
            let tool_failed = tool_results.len() - tool_passed;
            println!(
                "  Tool tests: {} passed, {} failed",
                tool_passed.to_string().green(),
                if tool_failed > 0 {
                    tool_failed.to_string().red()
                } else {
                    tool_failed.to_string().dimmed()
                }
            );
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
    println!("\n{}", "=== Execution Results ===".bold().cyan());
    println!(
        "  {} {}",
        "Total scenarios:".dimmed(),
        result.total_scenarios
    );
    println!(
        "  {} {}",
        "Passed:".dimmed(),
        result.passed.to_string().bold().green()
    );
    println!(
        "  {} {}",
        "Failed:".dimmed(),
        if result.failed > 0 {
            result.failed.to_string().bold().red()
        } else {
            result.failed.to_string().dimmed()
        }
    );
    println!(
        "  {} {}",
        "Skipped:".dimmed(),
        if result.skipped > 0 {
            result.skipped.to_string().yellow()
        } else {
            result.skipped.to_string().dimmed()
        }
    );
    println!("  {} {}ms", "Duration:".dimmed(), result.duration_ms);
    let pass_rate = result.pass_rate();
    let rate_str = format!("{pass_rate:.1}%");
    let colored_rate = if pass_rate >= 90.0 {
        rate_str.green()
    } else if pass_rate >= 70.0 {
        rate_str.yellow()
    } else {
        rate_str.red()
    };
    println!("  {} {colored_rate}", "Pass rate:".dimmed());

    if let Some(ref gateway_fail) = result.gateway_failed {
        println!("  {} {gateway_fail}", "Gateway FAILED:".bold().red());
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
                println!(
                    "\n{} {}",
                    "Evidence saved to:".green(),
                    evidence_path.display().to_string().cyan()
                );
            }
        }
        Err(e) => eprintln!("Error serializing evidence: {e}"),
    }
}

/// Log environment information for fail-fast diagnostics (§12.5.3)
fn log_environment() {
    let tag = "[ENVIRONMENT]".dimmed().cyan();
    eprintln!("\n{tag} {}", "=== Diagnostic Context ===".dimmed());
    eprintln!(
        "{tag} {} {} {}",
        "OS:".dimmed(),
        std::env::consts::OS.dimmed(),
        std::env::consts::ARCH.dimmed()
    );
    eprintln!(
        "{tag} {} {}",
        "apr-qa version:".dimmed(),
        env!("CARGO_PKG_VERSION").dimmed()
    );

    // Git context
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let commit = String::from_utf8_lossy(&output.stdout);
            eprintln!(
                "{tag} {} {}",
                "Git commit:".dimmed(),
                commit.trim().dimmed()
            );
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout);
            eprintln!(
                "{tag} {} {}",
                "Git branch:".dimmed(),
                branch.trim().dimmed()
            );
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
                eprintln!(
                    "{tag} {} {}",
                    "Git dirty:".dimmed(),
                    format!("{dirty_count} file(s) modified").dimmed()
                );
            }
        }
    }

    // apr CLI version
    if let Ok(output) = std::process::Command::new("apr").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("{tag} {} {}", "apr-cli:".dimmed(), version.trim().dimmed());
        }
    }

    // Rust version
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("{tag} {}", version.trim().dimmed());
        }
    }

    eprintln!("{tag} {}\n", "===========================".dimmed());
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

// Certification workflow — see certification.rs
include!("certification.rs");

// Export and cache — see export.rs
include!("export.rs");
