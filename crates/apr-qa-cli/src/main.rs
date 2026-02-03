//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

use apr_qa_cli::{
    CertTier, PlaybookRunConfig, build_certification_config, build_execution_config,
    calculate_mqs_score, calculate_popperian_score, collect_evidence, execute_auto_tickets,
    execute_playbook, filter_models_by_size, generate_html_report, generate_junit_report,
    generate_lock_file, generate_model_scenarios, generate_tickets_from_evidence, list_all_models,
    load_playbook, parse_evidence, parse_failure_policy, playbook_path_for_model,
    scenarios_to_json, scenarios_to_yaml,
};
use apr_qa_runner::ToolExecutor;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    }
}

#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
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

    // Validate failure policy
    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }

    println!("Running playbook: {}", playbook.name);
    println!("  Total tests: {}", playbook.total_tests());
    println!("  Dry run: {dry_run}");
    if let Some(ref path) = model_path {
        println!("  Model path: {path}");
    }
    println!("  Workers: {workers}");
    println!("  Timeout: {timeout}ms");

    let run_config = PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers,
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

    match execute_playbook(&playbook, config) {
        Ok(result) => {
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

            // Create output directory
            if let Err(e) = std::fs::create_dir_all(output_dir) {
                eprintln!("Error creating output directory: {e}");
                return;
            }

            // Save evidence
            let evidence_path = output_dir.join("evidence.json");
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
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
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

    // Calculate scores
    let mqs_score = match calculate_mqs_score(model_id, &collector) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let popperian_score = calculate_popperian_score(model_id, &collector);

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    let gen_html = formats == "all" || formats == "html";
    let gen_junit = formats == "all" || formats == "junit";

    if gen_html {
        match generate_html_report(
            &format!("MQS Report: {model_id}"),
            &mqs_score,
            &popperian_score,
            &collector,
        ) {
            Ok(html) => {
                let path = output_dir.join("report.html");
                if let Err(e) = std::fs::write(&path, html) {
                    eprintln!("Error writing HTML report: {e}");
                } else {
                    println!("HTML report: {}", path.display());
                }
            }
            Err(e) => eprintln!("{e}"),
        }
    }

    if gen_junit {
        match generate_junit_report(model_id, &collector, &mqs_score) {
            Ok(xml) => {
                let path = output_dir.join("junit.xml");
                if let Err(e) = std::fs::write(&path, xml) {
                    eprintln!("Error writing JUnit report: {e}");
                } else {
                    println!("JUnit report: {}", path.display());
                }
            }
            Err(e) => eprintln!("{e}"),
        }
    }

    // Save MQS score as JSON
    let score_path = output_dir.join("mqs.json");
    match serde_json::to_string_pretty(&mqs_score) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&score_path, json) {
                eprintln!("Error writing MQS JSON: {e}");
            } else {
                println!("MQS score: {}", score_path.display());
            }
        }
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
    use apr_qa_gen::{HfParityOracle, Tolerance, hash_prompt};

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

    // List mode: show available golden outputs
    if list {
        println!("\n=== Available Golden Outputs ===\n");
        let manifest_path = corpus_dir.join("manifest.json");
        if manifest_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(prompts) = manifest.get("prompts").and_then(|p| p.as_array()) {
                        println!("Found {} golden outputs:\n", prompts.len());
                        // List the JSON files to get the prompts
                        for entry in std::fs::read_dir(&corpus_dir)
                            .into_iter()
                            .flatten()
                            .flatten()
                        {
                            let path = entry.path();
                            if path.extension().is_some_and(|e| e == "json")
                                && path.file_stem().is_some_and(|s| s != "manifest")
                            {
                                if let Ok(json) = std::fs::read_to_string(&path) {
                                    if let Ok(meta) =
                                        serde_json::from_str::<serde_json::Value>(&json)
                                    {
                                        if let Some(prompt_str) =
                                            meta.get("prompt").and_then(|p| p.as_str())
                                        {
                                            let hash = path
                                                .file_stem()
                                                .map(|s| s.to_string_lossy())
                                                .unwrap_or_default();
                                            let truncated = if prompt_str.len() > 50 {
                                                format!("{}...", &prompt_str[..50])
                                            } else {
                                                prompt_str.to_string()
                                            };
                                            println!("  [{hash}] {truncated}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // Self-check mode: verify golden outputs against themselves
    if self_check {
        println!("\n=== Self-Check Mode ===");
        println!("Verifying golden outputs match themselves (sanity check)...\n");

        let mut passed = 0;
        let mut failed = 0;

        for entry in std::fs::read_dir(&corpus_dir)
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && path.file_stem().is_some_and(|s| s != "manifest")
            {
                if let Ok(json) = std::fs::read_to_string(&path) {
                    if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&json) {
                        if let Some(prompt_str) = meta.get("prompt").and_then(|p| p.as_str()) {
                            match oracle.load_golden(prompt_str) {
                                Ok(golden) => {
                                    // Verify against itself
                                    match oracle.tensors_close(&golden.logits, &golden.logits) {
                                        Ok(()) => {
                                            passed += 1;
                                            let short_prompt = if prompt_str.len() > 40 {
                                                format!("{}...", &prompt_str[..40])
                                            } else {
                                                prompt_str.to_string()
                                            };
                                            println!("  ✓ {short_prompt}");
                                        }
                                        Err(diff) => {
                                            failed += 1;
                                            eprintln!("  ✗ {prompt_str}: {diff}");
                                        }
                                    }
                                }
                                Err(e) => {
                                    failed += 1;
                                    eprintln!("  ✗ Failed to load {prompt_str}: {e}");
                                }
                            }
                        }
                    }
                }
            }
        }

        println!("\n=== Self-Check Results ===");
        println!("Passed: {passed}");
        println!("Failed: {failed}");

        if failed > 0 {
            std::process::exit(1);
        }
        return;
    }

    // Verification mode: compare logits file against golden
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

    // Load the logits to verify
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

    let logits: Vec<f32> = logits_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Load golden and compare
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
) {
    use apr_qa_certify::{
        CertificationStatus, CertificationTier, grade_from_tier, parse_csv, score_from_tier,
        status_from_tier, write_csv,
    };
    use chrono::Utc;

    // Parse tier
    let tier: CertTier = match tier_str.parse() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Default model_cache to ~/.cache/apr-models when not provided
    let model_cache: Option<PathBuf> = if model_cache.is_none() {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let default_cache = PathBuf::from(home).join(".cache/apr-models");
        println!("Auto-resolving model cache: {}", default_cache.display());
        Some(default_cache)
    } else {
        model_cache
    };

    println!("=== APR Model Certification ===\n");
    println!("Tier: {tier_str}");
    println!("Dry run: {dry_run}");
    if let Some(ref cache) = model_cache {
        println!("Model cache: {}", cache.display());
    }
    println!();

    // Load current certification data
    let csv_path = std::path::Path::new("docs/certifications/models.csv");
    let csv_content = match std::fs::read_to_string(csv_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading models.csv: {e}");
            std::process::exit(1);
        }
    };

    let mut certifications: Vec<apr_qa_certify::ModelCertification> = match parse_csv(&csv_content)
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing models.csv: {e}");
            std::process::exit(1);
        }
    };

    // Determine which models to certify
    let models_to_certify: Vec<String> = if all {
        certifications.iter().map(|c| c.model_id.clone()).collect()
    } else if let Some(ref fam) = family {
        certifications
            .iter()
            .filter(|c| c.family == *fam)
            .map(|c| c.model_id.clone())
            .collect()
    } else if !model_ids.is_empty() {
        model_ids.to_vec()
    } else {
        eprintln!("Error: Specify --all, --family, or model IDs");
        std::process::exit(1);
    };

    println!("Models to certify: {}\n", models_to_certify.len());

    if dry_run {
        for model_id in &models_to_certify {
            let playbook_name = playbook_path_for_model(model_id, tier);
            println!("  Would certify: {model_id}");
            println!("    Playbook: {playbook_name}");
        }
        return;
    }

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    // Certify each model
    let mut certified_count = 0;
    let mut failed_count = 0;

    for model_id in &models_to_certify {
        let short: &str = model_id.split('/').next_back().unwrap_or(model_id);
        let playbook_name = playbook_path_for_model(model_id, tier);

        println!("--- Certifying: {model_id} ---");
        println!("  Playbook: {playbook_name}");

        // Auto-populate model cache before execution
        if let Some(ref cache) = model_cache {
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

        // Configure execution
        let model_cache_path = model_cache.as_ref().map(|cache| {
            cache
                .join(short.to_lowercase().replace('.', "-"))
                .to_string_lossy()
                .to_string()
        });

        let config = build_certification_config(tier, model_cache_path);

        match execute_playbook(&playbook, config) {
            Ok(result) => {
                println!("  Scenarios: {}", result.total_scenarios);
                println!("  Passed: {}", result.passed);
                println!("  Failed: {}", result.failed);
                println!("  Pass rate: {:.1}%", result.pass_rate());

                // Calculate MQS score using tier-aware scoring
                let evidence_vec: Vec<_> = result.evidence.all().to_vec();
                let collector = collect_evidence(evidence_vec);
                let mqs = match calculate_mqs_score(model_id, &collector) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("  Error calculating MQS: {e}");
                        failed_count += 1;
                        continue;
                    }
                };

                // Determine certification tier from CLI tier
                let cert_tier = match tier {
                    CertTier::Mvp => CertificationTier::Mvp,
                    CertTier::Smoke | CertTier::Quick | CertTier::Standard | CertTier::Deep => {
                        CertificationTier::Full
                    }
                };

                // Use tier-aware scoring
                let pass_rate = result.pass_rate() / 100.0; // Convert percentage to 0-1
                let has_p0 = result.gateway_failed.is_some();
                let raw_score = score_from_tier(cert_tier, pass_rate, has_p0);
                let status = status_from_tier(cert_tier, pass_rate, has_p0);
                let grade = grade_from_tier(cert_tier, pass_rate, has_p0);

                println!("  Tier: {tier_str}");
                println!("  MQS Score: {raw_score}/1000");
                println!("  Grade: {grade}");
                println!("  Status: {status}");

                // Run 6-column profiling
                let mut profile = apr_qa_runner::SixColumnProfile::default();

                if let Some(ref cache) = model_cache {
                    // Model cache structure: <cache>/<model-short-name>/<format>/<file>
                    let model_dir = cache.join(short.to_lowercase().replace('.', "-"));

                    if model_dir.exists() {
                        println!("  Running 6-column profiling...");
                        match apr_qa_runner::run_six_column_profile(
                            apr_binary, &model_dir, 1, // warmup
                            2, // iterations
                        ) {
                            Ok(p) => {
                                profile = p;
                                // Print conversion results
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
                                        conv.source_format,
                                        conv.target_format,
                                        status,
                                        conv.duration_ms
                                    );
                                    if let Some(ref err) = conv.error {
                                        // Print first line of error
                                        if let Some(line) = err.lines().last() {
                                            println!("      {line}");
                                        }
                                    }
                                }
                                // Print throughput results
                                println!("    Throughput (tok/s):");
                                if let Some(tps) = profile.tps_gguf_cpu {
                                    println!("      GGUF CPU: {tps:.1}");
                                }
                                if let Some(tps) = profile.tps_gguf_gpu {
                                    println!("      GGUF GPU: {tps:.1}");
                                }
                                if let Some(tps) = profile.tps_apr_cpu {
                                    println!("      APR CPU:  {tps:.1}");
                                }
                                if let Some(tps) = profile.tps_apr_gpu {
                                    println!("      APR GPU:  {tps:.1}");
                                }
                                if let Some(tps) = profile.tps_st_cpu {
                                    println!("      ST CPU:   {tps:.1}");
                                }
                                if let Some(tps) = profile.tps_st_gpu {
                                    println!("      ST GPU:   {tps:.1}");
                                }
                                println!(
                                    "    Total profiling time: {}ms",
                                    profile.total_duration_ms
                                );

                                // Check assertions from playbook
                                if let Some(ref profile_ci) = playbook.profile_ci {
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
                            }
                            Err(e) => {
                                eprintln!("  Profiling failed: {e}");
                            }
                        }
                    }
                }

                // Update certification record
                if let Some(cert) = certifications.iter_mut().find(|c| c.model_id == *model_id) {
                    // Check if assertions failed - if so, block certification
                    let (final_status, final_grade, final_tier) =
                        if profile.failed_assertions.is_empty() {
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
                    // Set gateway status from MQS gateway results
                    let gw = &mqs.gateways;
                    cert.g1 = gw.first().is_some_and(|g| g.passed);
                    cert.g2 = gw.get(1).is_some_and(|g| g.passed);
                    cert.g3 = gw.get(2).is_some_and(|g| g.passed);
                    cert.g4 = gw.get(3).is_some_and(|g| g.passed);

                    // Set tok/s values from 6-column profiling
                    cert.tps_gguf_cpu = profile.tps_gguf_cpu;
                    cert.tps_gguf_gpu = profile.tps_gguf_gpu;
                    cert.tps_apr_cpu = profile.tps_apr_cpu;
                    cert.tps_apr_gpu = profile.tps_apr_gpu;
                    cert.tps_st_cpu = profile.tps_st_cpu;
                    cert.tps_st_gpu = profile.tps_st_gpu;
                }

                // Save evidence to model-specific directory
                let model_output = output_dir.join(short.to_lowercase().replace('.', "-"));
                if let Err(e) = std::fs::create_dir_all(&model_output) {
                    eprintln!("  Error creating model output dir: {e}");
                }

                let evidence_path = model_output.join("evidence.json");
                if let Ok(json) = result.evidence.to_json() {
                    let _ = std::fs::write(&evidence_path, json);
                    println!("  Evidence: {}", evidence_path.display());
                }

                certified_count += 1;
                println!();
            }
            Err(e) => {
                eprintln!("  Execution failed: {e}");
                failed_count += 1;
            }
        }
    }

    // Write updated CSV
    let csv_output = write_csv(&certifications);
    if let Err(e) = std::fs::write(csv_path, &csv_output) {
        eprintln!("Error writing models.csv: {e}");
    } else {
        println!("Updated: {}", csv_path.display());
    }

    // §3.1: Integrity check warning
    if !no_integrity_check {
        let lock_path = "playbooks/playbook.lock.yaml";
        if !std::path::Path::new(lock_path).exists() {
            eprintln!(
                "[WARN] No playbook lock file found at {lock_path}. Run `apr-qa lock-playbooks` to generate one."
            );
        }
    }

    // §3.6: Auto-ticket generation from failures
    if auto_ticket {
        // Collect all evidence files from output directory
        let mut all_evidence: Vec<apr_qa_runner::Evidence> = Vec::new();
        for model_id in &models_to_certify {
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

        if !all_evidence.is_empty() {
            let tickets = execute_auto_tickets(&all_evidence, ticket_repo);
            if tickets.is_empty() {
                println!(
                    "\n[AUTO-TICKET] No structured tickets generated (no classified failures)."
                );
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
    }

    println!("\n=== Certification Summary ===");
    println!("Certified: {certified_count}");
    println!("Failed: {failed_count}");
    println!("Total: {}", models_to_certify.len());
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

    // Skip if already populated (gguf dir has a .gguf file)
    if gguf_dir.exists() && has_file_with_ext(&gguf_dir, "gguf") {
        println!("  Cache already populated: {}", model_dir.display());
        return;
    }

    println!("  Auto-populating model cache...");

    // Create subdirectories
    for dir in [&gguf_dir, &apr_dir, &st_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("  Error creating {}: {e}", dir.display());
            return;
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let home = std::path::Path::new(&home);

    // Step 1: Pull model via apr (ensures it's in pacha cache)
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

    // Step 2: Find GGUF in pacha cache via manifest
    let manifest_path = home.join(".cache/pacha/models/manifest.json");
    if let Some(gguf_path) = find_gguf_in_pacha(&manifest_path, model_id) {
        let link = gguf_dir.join("model.gguf");
        if !link.exists() {
            match std::os::unix::fs::symlink(&gguf_path, &link) {
                Ok(()) => println!("  Linked GGUF: {gguf_path}"),
                Err(e) => eprintln!("  Error symlinking GGUF: {e}"),
            }
        }
    } else {
        eprintln!("  No GGUF found in pacha cache for {model_id}");
    }

    // Step 3: Find SafeTensors in HF cache
    let (org, repo) = split_model_id(model_id);
    let hf_model_dir = home
        .join(".cache/huggingface/hub")
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");

    if let Some(st_path) = find_safetensors_in_hf(&hf_model_dir) {
        let link = st_dir.join("model.safetensors");
        if !link.exists() {
            match std::os::unix::fs::symlink(&st_path, &link) {
                Ok(()) => println!("  Linked SafeTensors: {}", st_path.display()),
                Err(e) => eprintln!("  Error symlinking SafeTensors: {e}"),
            }
        }

        // Copy config.json from the same snapshot directory
        if let Some(snapshot_dir) = st_path.parent() {
            let config_src = snapshot_dir.join("config.json");
            let config_dst = st_dir.join("config.json");
            if config_src.exists() && !config_dst.exists() {
                match std::fs::copy(&config_src, &config_dst) {
                    Ok(_) => println!("  Copied config.json"),
                    Err(e) => eprintln!("  Error copying config.json: {e}"),
                }
            }
        }
    } else {
        eprintln!("  No SafeTensors found in HF cache for {model_id}");
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
