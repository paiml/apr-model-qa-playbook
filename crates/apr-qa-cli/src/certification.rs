#[allow(clippy::too_many_lines)]
#[allow(clippy::fn_params_excessive_bools)]
fn run_certification(
    all: bool,
    family: Option<String>,
    tier_str: &str,
    kernel_class: Option<String>,
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
    let models_to_certify = resolve_models_for_certification(
        all,
        family.as_deref(),
        model_ids,
        kernel_class.as_deref(),
        &certifications,
    );

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
        println!(
            "{} {}",
            "Updated:".green(),
            csv_path.display().to_string().cyan()
        );
    }

    warn_missing_lock_file(no_integrity_check);

    if auto_ticket {
        run_auto_ticket_generation(&models_to_certify, output_dir, ticket_repo);
    }

    println!("\n{}", "=== Certification Summary ===".bold().cyan());
    println!(
        "{} {}",
        "Certified:".dimmed(),
        certified_count.to_string().bold().green()
    );
    println!(
        "{} {}",
        "Failed:".dimmed(),
        if failed_count > 0 {
            failed_count.to_string().bold().red()
        } else {
            failed_count.to_string().dimmed()
        }
    );
    println!("{} {}", "Total:".dimmed(), models_to_certify.len());
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
    println!("{}\n", "=== APR Model Certification ===".bold().cyan());
    println!("{} {}", "Tier:".dimmed(), tier_str.bold().magenta());
    if dry_run {
        println!("{} {}", "Dry run:".dimmed(), "true".yellow());
    } else {
        println!("{} {}", "Dry run:".dimmed(), "false".dimmed());
    }
    println!("{} {fail_fast}", "Fail-fast:".dimmed());
    if let Some(cache) = model_cache {
        println!(
            "{} {}",
            "Model cache:".dimmed(),
            cache.display().to_string().cyan()
        );
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

fn resolve_models_for_certification(
    all: bool,
    family: Option<&str>,
    model_ids: &[String],
    kernel_class: Option<&str>,
    certifications: &[apr_qa_certify::ModelCertification],
) -> Vec<String> {
    kernel_class.map_or_else(
        || determine_models_to_certify(all, family, model_ids, certifications),
        |kc_str| {
            let kc: apr_qa_gen::KernelClass = match kc_str.parse() {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };
            let families_and_ids: Vec<(String, String)> = certifications
                .iter()
                .map(|c| (c.family.clone(), c.model_id.clone()))
                .collect();
            let models = apr_qa_gen::models_in_class(kc, &families_and_ids);
            if models.is_empty() {
                eprintln!("No models found for kernel class {kc}");
                std::process::exit(1);
            }
            println!(
                "Kernel class {kc} ({}) — {} models, proof: {}",
                kc.label(),
                models.len(),
                kc.representative_model(),
            );
            models
        },
    )
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

        println!(
            "{} {} {}",
            "---".bold(),
            format!("Certifying: {model_id}").bold(),
            "---".bold()
        );
        println!("  {} {playbook_name}", "Playbook:".dimmed());

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

            print_certification_scores(tier_str, raw_score, &grade, status);

            let profile = if matches!(tier, CertTier::DimensionalSmoke) {
                apr_qa_runner::SixColumnProfile::default()
            } else {
                run_profiling_phase(&result, playbook, model_cache, short, apr_binary, fail_fast)
            };

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

fn print_certification_scores(
    tier_str: &str,
    raw_score: u32,
    grade: &str,
    status: apr_qa_certify::CertificationStatus,
) {
    println!("  {} {tier_str}", "Tier:".dimmed());
    let score_str = format!("{raw_score}/1000");
    let colored_score = if raw_score >= 700 {
        score_str.bold().green()
    } else if raw_score >= 400 {
        score_str.bold().yellow()
    } else {
        score_str.bold().red()
    };
    println!("  {} {colored_score}", "MQS Score:".dimmed());
    let colored_grade = match grade {
        "A" | "B" => grade.green(),
        "C" | "D" => grade.yellow(),
        _ => grade.red(),
    };
    println!("  {} {colored_grade}", "Grade:".dimmed());
    let status_str = format!("{status}");
    let colored_status = if status_str.contains("Certified") || status_str.contains("Passed") {
        status_str.green()
    } else {
        status_str.red()
    };
    println!("  {} {colored_status}", "Status:".dimmed());
}

fn print_execution_summary(result: &apr_qa_runner::ExecutionResult) {
    println!("  {} {}", "Scenarios:".dimmed(), result.total_scenarios);
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
        CertTier::DimensionalSmoke
        | CertTier::Smoke
        | CertTier::Quick
        | CertTier::Standard
        | CertTier::Deep => CertificationTier::Full,
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
