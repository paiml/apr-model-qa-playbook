//! APR QA CLI
//!
//! Command-line interface for running model qualification playbooks.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]

use apr_qa_gen::{ModelId, ModelRegistry, ScenarioGenerator};
use apr_qa_report::{
    html::HtmlDashboard, junit::JunitReport, mqs::MqsCalculator, popperian::PopperianCalculator,
    ticket::TicketGenerator,
};
use apr_qa_runner::{ExecutionConfig, Executor, FailurePolicy, Playbook};
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

#[derive(Subcommand)]
enum Commands {
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
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
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
            );
        }
        Commands::Tools {
            model_path,
            no_gpu,
            output,
        } => {
            run_tool_tests(&model_path, no_gpu, &output);
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
        } => {
            generate_tickets(&evidence, &repo, black_swans_only, min_occurrences);
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
) {
    println!("Loading playbook: {}", playbook_path.display());

    let playbook = match Playbook::from_file(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error loading playbook: {e}");
            std::process::exit(1);
        }
    };

    let policy = match failure_policy {
        "stop-on-first" => FailurePolicy::StopOnFirst,
        "stop-on-p0" => FailurePolicy::StopOnP0,
        "collect-all" => FailurePolicy::CollectAll,
        _ => {
            eprintln!("Unknown failure policy: {failure_policy}");
            std::process::exit(1);
        }
    };

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

    let config = ExecutionConfig {
        failure_policy: policy,
        dry_run,
        max_workers: workers,
        subprocess_mode: subprocess,
        model_path,
        default_timeout_ms: timeout,
        no_gpu,
        run_conversion_tests: !skip_conversion_tests,
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

    let mut executor = Executor::with_config(config);

    // Run tool tests if enabled
    if run_tool_tests_flag && subprocess {
        if let Some(ref mp) = executor.config().model_path {
            use apr_qa_runner::ToolExecutor;
            println!("\n=== Running APR Tool Tests ===");
            let tool_executor = ToolExecutor::new(mp.clone(), no_gpu, timeout);
            let tool_results = tool_executor.execute_all();
            let tool_passed = tool_results.iter().filter(|r| r.passed).count();
            let tool_failed = tool_results.len() - tool_passed;
            println!("  Tool tests: {tool_passed} passed, {tool_failed} failed");
        }
    }

    match executor.execute(&playbook) {
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
            eprintln!("Execution failed: {e}");
            std::process::exit(1);
        }
    }
}

fn generate_scenarios(model_id: &str, count: usize, format: &str) {
    let parts: Vec<&str> = model_id.split('/').collect();
    let (org, name) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("unknown", model_id)
    };

    let model = ModelId::new(org, name);
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(count);

    let scenarios = generator.generate();

    println!("Generated {} scenarios for {model_id}", scenarios.len());

    match format {
        "yaml" => {
            for scenario in &scenarios {
                match serde_yaml::to_string(scenario) {
                    Ok(yaml) => println!("---\n{yaml}"),
                    Err(e) => eprintln!("Error serializing scenario: {e}"),
                }
            }
        }
        "json" => match serde_json::to_string_pretty(&scenarios) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("Error serializing scenarios: {e}"),
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

    let evidence: Vec<apr_qa_runner::Evidence> = match serde_json::from_str(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error parsing evidence JSON: {e}");
            std::process::exit(1);
        }
    };

    let mut collector = apr_qa_runner::EvidenceCollector::new();
    for e in evidence {
        collector.add(e);
    }

    let calculator = MqsCalculator::new();
    match calculator.calculate(model_id, &collector) {
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
            eprintln!("Error calculating score: {e}");
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

    let evidence: Vec<apr_qa_runner::Evidence> = match serde_json::from_str(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error parsing evidence JSON: {e}");
            std::process::exit(1);
        }
    };

    let mut collector = apr_qa_runner::EvidenceCollector::new();
    for e in evidence {
        collector.add(e);
    }

    // Calculate scores
    let mqs_calculator = MqsCalculator::new();
    let mqs_score = match mqs_calculator.calculate(model_id, &collector) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error calculating MQS: {e}");
            std::process::exit(1);
        }
    };

    let popperian_calculator = PopperianCalculator::new();
    let popperian_score = popperian_calculator.calculate(model_id, &collector);

    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    let generate_html = formats == "all" || formats == "html";
    let generate_junit = formats == "all" || formats == "junit";

    if generate_html {
        let dashboard = HtmlDashboard::new(format!("MQS Report: {model_id}"));
        match dashboard.generate(&mqs_score, &popperian_score, &collector) {
            Ok(html) => {
                let path = output_dir.join("report.html");
                if let Err(e) = std::fs::write(&path, html) {
                    eprintln!("Error writing HTML report: {e}");
                } else {
                    println!("HTML report: {}", path.display());
                }
            }
            Err(e) => eprintln!("Error generating HTML: {e}"),
        }
    }

    if generate_junit {
        let junit = JunitReport::new(model_id);
        match junit.generate(&collector, &mqs_score) {
            Ok(xml) => {
                let path = output_dir.join("junit.xml");
                if let Err(e) = std::fs::write(&path, xml) {
                    eprintln!("Error writing JUnit report: {e}");
                } else {
                    println!("JUnit report: {}", path.display());
                }
            }
            Err(e) => eprintln!("Error generating JUnit: {e}"),
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
    let registry = ModelRegistry::with_defaults();
    let models = registry.all();

    println!("=== Available Models ===\n");

    for model in models {
        if let Some(filter) = size_filter {
            let size_str = format!("{:?}", model.size).to_lowercase();
            if !size_str.contains(&filter.to_lowercase()) {
                continue;
            }
        }

        println!("  {} ({:?})", model.id.hf_repo(), model.size);
    }
}

fn generate_tickets(
    evidence_path: &PathBuf,
    repo: &str,
    black_swans_only: bool,
    min_occurrences: usize,
) {
    let evidence_json = match std::fs::read_to_string(evidence_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading evidence file: {e}");
            std::process::exit(1);
        }
    };

    let evidence: Vec<apr_qa_runner::Evidence> = match serde_json::from_str(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error parsing evidence JSON: {e}");
            std::process::exit(1);
        }
    };

    let mut generator = TicketGenerator::new(repo).with_min_occurrences(min_occurrences);

    if black_swans_only {
        generator = generator.black_swans_only();
    }

    let tickets = generator.generate_from_evidence(&evidence);

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

fn run_tool_tests(model_path: &std::path::Path, no_gpu: bool, output_dir: &std::path::Path) {
    use apr_qa_runner::ToolExecutor;

    println!("=== APR Tool Coverage Tests ===\n");
    println!("Model: {}", model_path.display());
    println!("GPU: {}\n", if no_gpu { "disabled" } else { "enabled" });

    let executor = ToolExecutor::new(model_path.to_string_lossy().to_string(), no_gpu, 120_000);

    let results = executor.execute_all();

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
