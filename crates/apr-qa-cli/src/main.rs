//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

use apr_qa_cli::{
    PlaybookRunConfig, build_execution_config, calculate_mqs_score, calculate_popperian_score,
    collect_evidence, execute_playbook, filter_models_by_size, generate_html_report,
    generate_junit_report, generate_model_scenarios, generate_tickets_from_evidence,
    list_all_models, load_playbook, parse_evidence, parse_failure_policy, scenarios_to_json,
    scenarios_to_yaml,
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

/// Certification tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CertTier {
    /// Tier 1: Smoke test (~1-2 min per model)
    Smoke,
    /// Tier 2: MVP - all formats/backends/modalities (~5-10 min per model)
    Mvp,
    /// Tier 3: Quick check (~10-30 min per model)
    #[default]
    Quick,
    /// Tier 4: Standard certification (~1-2 hr per model)
    Standard,
    /// Tier 5: Deep certification (~8-24 hr per model)
    Deep,
}

impl std::str::FromStr for CertTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "mvp" => Ok(Self::Mvp),
            "quick" => Ok(Self::Quick),
            "standard" => Ok(Self::Standard),
            "deep" => Ok(Self::Deep),
            _ => Err(format!(
                "Unknown tier: {s}. Use: smoke, mvp, quick, standard, deep"
            )),
        }
    }
}

impl CertTier {
    const fn playbook_suffix(self) -> &'static str {
        match self {
            Self::Smoke => "-smoke",
            Self::Mvp => "-mvp",
            Self::Quick => "-quick",
            Self::Standard | Self::Deep => "",
        }
    }
}

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

        /// Enable real subprocess execution (requires --model-cache)
        #[arg(long)]
        subprocess: bool,
    },

    /// Run a playbook
    Run {
        /// Path to playbook YAML file
        #[arg(value_name = "PLAYBOOK")]
        playbook: PathBuf,

        /// Output directory for reports
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Failure policy (stop-on-first, stop-on-p0, collect-all)
        #[arg(long, default_value = "stop-on-p0")]
        failure_policy: String,

        /// Dry run (don't execute, just show what would be done)
        #[arg(long)]
        dry_run: bool,

        /// Maximum parallel workers
        #[arg(long, default_value = "4")]
        workers: usize,

        /// Enable subprocess mode (run actual apr commands)
        #[arg(long)]
        subprocess: bool,

        /// Path to model file (required for subprocess mode)
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
            subprocess,
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
                subprocess,
            );
        }
        Commands::Run {
            playbook,
            output,
            failure_policy,
            dry_run,
            workers,
            subprocess,
            model_path,
            timeout,
            no_gpu,
            skip_conversion_tests,
            run_tool_tests,
            profile_ci,
            no_differential,
            no_trace_payload,
        } => {
            run_playbook(
                &playbook,
                &output,
                &failure_policy,
                dry_run,
                workers,
                subprocess,
                model_path,
                timeout,
                no_gpu,
                skip_conversion_tests,
                run_tool_tests,
                profile_ci,
                no_differential,
                no_trace_payload,
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
    }
}

#[allow(clippy::fn_params_excessive_bools)]
fn run_playbook(
    playbook_path: &PathBuf,
    output_dir: &PathBuf,
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    subprocess: bool,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests_flag: bool,
    profile_ci: bool,
    no_differential: bool,
    no_trace_payload: bool,
) {
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

    // Validate subprocess mode requirements
    if subprocess && model_path.is_none() {
        eprintln!("Error: --model-path is required when using --subprocess mode");
        std::process::exit(1);
    }

    println!("Running playbook: {}", playbook.name);
    println!("  Total tests: {}", playbook.total_tests());
    println!("  Dry run: {dry_run}");
    println!("  Subprocess mode: {subprocess}");
    if let Some(ref path) = model_path {
        println!("  Model path: {path}");
    }
    println!("  Workers: {workers}");
    println!("  Timeout: {timeout}ms");

    let run_config = PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers,
        subprocess,
        model_path: model_path.clone(),
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests: run_tool_tests_flag,
        run_differential_tests: !no_differential,
        run_profile_ci: profile_ci,
        run_trace_payload: !no_trace_payload,
    };

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Print conversion test status (P0 CRITICAL)
    if !skip_conversion_tests && subprocess {
        println!("  Conversion tests: ENABLED (P0 CRITICAL)");
    } else if skip_conversion_tests {
        println!("  Conversion tests: DISABLED (WARNING: P0 tests skipped)");
    }

    // Print tool test status
    if run_tool_tests_flag && subprocess {
        println!("  Tool tests: ENABLED");
    }

    // Run tool tests if enabled
    if run_tool_tests_flag && subprocess {
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
    subprocess: bool,
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

    // Validate subprocess requirements
    if subprocess && model_cache.is_none() {
        eprintln!("Error: --model-cache is required when using --subprocess mode");
        std::process::exit(1);
    }

    println!("=== APR Model Certification ===\n");
    println!("Tier: {tier_str}");
    println!("Dry run: {dry_run}");
    println!("Subprocess mode: {subprocess}");
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

    // Helper to generate playbook path from model ID
    let playbook_path_for = |model_id: &str, tier: &CertTier| -> String {
        let short = model_id.split('/').next_back().unwrap_or(model_id);
        // Convert Qwen2.5-Coder-0.5B-Instruct -> qwen2.5-coder-0.5b
        let base = short
            .to_lowercase()
            .replace("-instruct", "")
            .replace("-it", "");
        format!(
            "playbooks/models/{}{}.playbook.yaml",
            base,
            tier.playbook_suffix()
        )
    };

    if dry_run {
        for model_id in &models_to_certify {
            let playbook_name = playbook_path_for(model_id, &tier);
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
        let playbook_name = playbook_path_for(model_id, &tier);

        println!("--- Certifying: {model_id} ---");
        println!("  Playbook: {playbook_name}");

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

        // Configure execution - use subprocess mode if enabled
        // Pass model cache directory (containing gguf/apr/safetensors subdirs)
        let model_cache_path = if subprocess {
            model_cache.as_ref().map(|cache| {
                cache
                    .join(short.to_lowercase().replace('.', "-"))
                    .to_string_lossy()
                    .to_string()
            })
        } else {
            None
        };

        let config = apr_qa_runner::ExecutionConfig {
            failure_policy: apr_qa_runner::FailurePolicy::CollectAll,
            dry_run: false,
            max_workers: 4,
            subprocess_mode: subprocess,
            model_path: model_cache_path,
            default_timeout_ms: 60000,
            no_gpu: false,                    // Allow GPU when in subprocess mode
            run_conversion_tests: subprocess, // Run P0 conversion tests in subprocess mode
            run_differential_tests: false,
            run_profile_ci: false,
            run_trace_payload: false,
            run_golden_rule_test: subprocess, // Run golden rule test in subprocess mode
            golden_reference_path: None,
        };

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

                // Run 6-column profiling if subprocess mode enabled
                let mut profile = apr_qa_runner::SixColumnProfile::default();

                if subprocess {
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

    println!("\n=== Certification Summary ===");
    println!("Certified: {certified_count}");
    println!("Failed: {failed_count}");
    println!("Total: {}", models_to_certify.len());
}
