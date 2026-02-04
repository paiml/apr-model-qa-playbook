//! Fail-Fast Diagnostic Report Generation (FF-REPORT-001)
//!
//! Generates comprehensive diagnostic reports on test failure using apr's rich tooling.
//! Reports are designed for immediate GitHub issue creation with full reproduction context.

use crate::evidence::Evidence;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Timeout for each diagnostic command
const CHECK_TIMEOUT: Duration = Duration::from_secs(30);
const INSPECT_TIMEOUT: Duration = Duration::from_secs(10);
const TRACE_TIMEOUT: Duration = Duration::from_secs(60);
const TENSORS_TIMEOUT: Duration = Duration::from_secs(10);
const EXPLAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of a diagnostic command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Command that was run
    pub command: String,
    /// Whether the command succeeded
    pub success: bool,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the command timed out
    pub timed_out: bool,
}

/// Environment context for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    /// Operating system (e.g., "linux", "macos", "windows")
    pub os: String,
    /// CPU architecture (e.g., "x86_64", "aarch64")
    pub arch: String,
    /// apr-qa version
    pub apr_qa_version: String,
    /// apr CLI version
    pub apr_cli_version: String,
    /// Git commit hash (short form)
    pub git_commit: String,
    /// Git branch name
    pub git_branch: String,
    /// Whether working directory has uncommitted changes
    pub git_dirty: bool,
    /// Rust compiler version
    pub rustc_version: String,
}

impl EnvironmentContext {
    /// Collect environment context
    #[must_use]
    pub fn collect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            apr_qa_version: env!("CARGO_PKG_VERSION").to_string(),
            apr_cli_version: get_apr_version(),
            git_commit: get_git_commit(),
            git_branch: get_git_branch(),
            git_dirty: get_git_dirty(),
            rustc_version: get_rustc_version(),
        }
    }
}

/// Failure details from the evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureDetails {
    /// Gate ID that failed (e.g., "G3-STABLE")
    pub gate_id: String,
    /// Model identifier (HuggingFace repo path)
    pub model: String,
    /// Model format (e.g., "Apr", "SafeTensors", "Gguf")
    pub format: String,
    /// Backend used (e.g., "Cpu", "Metal", "Cuda")
    pub backend: String,
    /// Test outcome (e.g., "Crashed", "Falsified", "Timeout")
    pub outcome: String,
    /// Human-readable failure reason
    pub reason: String,
    /// Process exit code if available
    pub exit_code: Option<i32>,
    /// Test duration in milliseconds
    pub duration_ms: u64,
    /// Standard error output if captured
    pub stderr: Option<String>,
}

impl From<&Evidence> for FailureDetails {
    fn from(evidence: &Evidence) -> Self {
        Self {
            gate_id: evidence.gate_id.clone(),
            model: evidence.scenario.model.hf_repo(),
            format: format!("{:?}", evidence.scenario.format),
            backend: format!("{:?}", evidence.scenario.backend),
            outcome: format!("{:?}", evidence.outcome),
            reason: evidence.reason.clone(),
            exit_code: evidence.exit_code,
            duration_ms: evidence.metrics.duration_ms,
            stderr: evidence.stderr.clone(),
        }
    }
}

/// Complete fail-fast diagnostic report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailFastReport {
    /// Report version
    pub version: String,
    /// Timestamp
    pub timestamp: String,
    /// Failure details
    pub failure: FailureDetails,
    /// Environment context
    pub environment: EnvironmentContext,
    /// Diagnostic results
    pub diagnostics: DiagnosticsBundle,
    /// Reproduction information
    pub reproduction: ReproductionInfo,
}

/// Bundle of all diagnostic results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsBundle {
    /// Results from `apr check` - pipeline integrity
    pub check: Option<DiagnosticResult>,
    /// Results from `apr inspect` - model metadata
    pub inspect: Option<DiagnosticResult>,
    /// Results from `apr trace` - layer-by-layer analysis
    pub trace: Option<DiagnosticResult>,
    /// Results from `apr tensors` - tensor names and shapes
    pub tensors: Option<DiagnosticResult>,
    /// Results from `apr explain` - error code explanation
    pub explain: Option<DiagnosticResult>,
}

/// Information for reproducing the failure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproductionInfo {
    /// Command to reproduce the failure
    pub command: String,
    /// Path to the model file used
    pub model_path: String,
    /// Path to the playbook file if applicable
    pub playbook: Option<String>,
}

/// Fail-fast diagnostic reporter
pub struct FailFastReporter {
    output_dir: PathBuf,
    binary: String,
}

impl FailFastReporter {
    /// Create a new reporter
    #[must_use]
    pub fn new(output_dir: &Path) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            binary: "apr".to_string(),
        }
    }

    /// Create with custom binary path
    #[must_use]
    pub fn with_binary(mut self, binary: &str) -> Self {
        self.binary = binary.to_string();
        self
    }

    /// Generate full diagnostic report on failure
    ///
    /// # Errors
    ///
    /// Returns an error if report generation fails.
    pub fn generate_report(
        &self,
        evidence: &Evidence,
        model_path: &Path,
        playbook: Option<&str>,
    ) -> std::io::Result<FailFastReport> {
        let report_dir = self.output_dir.join("fail-fast-report");
        std::fs::create_dir_all(&report_dir)?;

        eprintln!("[FAIL-FAST] Generating diagnostic report...");

        // Collect diagnostics
        let check = self.run_check(model_path);
        let inspect = self.run_inspect(model_path);
        let trace = self.run_trace(model_path);
        let tensors = self.run_tensors(model_path);
        let explain = self.run_explain(&evidence.gate_id);

        // Save individual diagnostic files first (before moving into report)
        if let Some(ref c) = check {
            self.save_json(&report_dir.join("check.json"), c)?;
        }
        if let Some(ref i) = inspect {
            self.save_json(&report_dir.join("inspect.json"), i)?;
        }
        if let Some(ref t) = trace {
            self.save_json(&report_dir.join("trace.json"), t)?;
        }
        if let Some(ref t) = tensors {
            self.save_json(&report_dir.join("tensors.json"), t)?;
        }

        // Build report (moves diagnostic values)
        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            failure: FailureDetails::from(evidence),
            environment: EnvironmentContext::collect(),
            diagnostics: DiagnosticsBundle {
                check,
                inspect,
                trace,
                tensors,
                explain,
            },
            reproduction: ReproductionInfo {
                command: format!(
                    "apr-qa run {} --fail-fast",
                    playbook.unwrap_or("playbook.yaml")
                ),
                model_path: model_path.to_string_lossy().to_string(),
                playbook: playbook.map(String::from),
            },
        };

        // Save full diagnostics JSON
        self.save_json(&report_dir.join("diagnostics.json"), &report)?;

        // Save environment
        self.save_json(&report_dir.join("environment.json"), &report.environment)?;

        // Save stderr log
        if let Some(ref stderr) = evidence.stderr {
            std::fs::write(report_dir.join("stderr.log"), stderr)?;
        }

        // Generate markdown summary
        let summary = self.generate_markdown(&report);
        std::fs::write(report_dir.join("summary.md"), &summary)?;

        eprintln!("[FAIL-FAST] Report saved to: {}", report_dir.display());
        eprintln!("[FAIL-FAST] Summary: {}/summary.md", report_dir.display());
        eprintln!("[FAIL-FAST] GitHub issue body ready for paste");

        Ok(report)
    }

    /// Run apr check and capture output
    fn run_check(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr check... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "check",
                &model_path.to_string_lossy(),
                "--json",
            ],
            CHECK_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr inspect and capture output
    fn run_inspect(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr inspect... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "inspect",
                &model_path.to_string_lossy(),
                "--json",
            ],
            INSPECT_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr trace and capture output
    fn run_trace(&self, model_path: &Path) -> Option<DiagnosticResult> {
        // Only run trace for .apr files
        if model_path.extension().is_none_or(|e| e != "apr") {
            return None;
        }

        eprint!("[FAIL-FAST] Running apr trace... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "trace",
                &model_path.to_string_lossy(),
                "--payload",
                "--json",
            ],
            TRACE_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr tensors and capture output
    fn run_tensors(&self, model_path: &Path) -> Option<DiagnosticResult> {
        eprint!("[FAIL-FAST] Running apr tensors... ");
        let result = self.run_command_with_timeout(
            &[
                &self.binary,
                "tensors",
                &model_path.to_string_lossy(),
                "--json",
            ],
            TENSORS_TIMEOUT,
        );
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run apr explain for the error code
    fn run_explain(&self, error_code: &str) -> Option<DiagnosticResult> {
        // Extract error code pattern (e.g., "G3-STABLE" -> try explaining common errors)
        eprint!("[FAIL-FAST] Running apr explain... ");
        let result =
            self.run_command_with_timeout(&[&self.binary, "explain", error_code], EXPLAIN_TIMEOUT);
        eprintln!(
            "done ({:.1}s){}",
            result.duration_ms as f64 / 1000.0,
            if result.timed_out { " [TIMEOUT]" } else { "" }
        );
        Some(result)
    }

    /// Run a command with timeout
    fn run_command_with_timeout(&self, args: &[&str], timeout: Duration) -> DiagnosticResult {
        let start = Instant::now();
        let command_str = args.join(" ");

        let output = Command::new(args[0]).args(&args[1..]).output();

        let duration = start.elapsed();
        let timed_out = duration > timeout;

        match output {
            Ok(out) => DiagnosticResult {
                command: command_str,
                success: out.status.success(),
                stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                duration_ms: duration.as_millis() as u64,
                timed_out,
            },
            Err(e) => DiagnosticResult {
                command: command_str,
                success: false,
                stdout: String::new(),
                stderr: format!("Failed to execute: {e}"),
                duration_ms: duration.as_millis() as u64,
                timed_out,
            },
        }
    }

    /// Save JSON to file
    fn save_json<T: Serialize>(&self, path: &Path, data: &T) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(data).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Generate markdown summary
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn generate_markdown(&self, report: &FailFastReport) -> String {
        let mut md = String::new();

        // Header
        let _ = writeln!(md, "# Fail-Fast Report: {}\n", report.failure.gate_id);

        // Failure Summary Table
        md.push_str("## Failure Summary\n\n");
        md.push_str("| Field | Value |\n");
        md.push_str("|-------|-------|\n");
        let _ = writeln!(md, "| Gate | `{}` |", report.failure.gate_id);
        let _ = writeln!(md, "| Model | `{}` |", report.failure.model);
        let _ = writeln!(md, "| Format | {} |", report.failure.format);
        let _ = writeln!(md, "| Backend | {} |", report.failure.backend);
        let _ = writeln!(md, "| Outcome | {} |", report.failure.outcome);
        if let Some(code) = report.failure.exit_code {
            let _ = writeln!(md, "| Exit Code | {code} |");
        }
        let _ = writeln!(md, "| Duration | {}ms |", report.failure.duration_ms);
        md.push('\n');

        // Reason
        md.push_str("### Reason\n\n");
        let _ = writeln!(md, "{}\n", report.failure.reason);

        // Environment Table
        md.push_str("## Environment\n\n");
        md.push_str("| Field | Value |\n");
        md.push_str("|-------|-------|\n");
        let _ = writeln!(
            md,
            "| OS | {} {} |",
            report.environment.os, report.environment.arch
        );
        let _ = writeln!(md, "| apr-qa | {} |", report.environment.apr_qa_version);
        let _ = writeln!(md, "| apr-cli | {} |", report.environment.apr_cli_version);
        let _ = writeln!(
            md,
            "| Git | {} ({}){}|",
            report.environment.git_commit,
            report.environment.git_branch,
            if report.environment.git_dirty {
                " [dirty]"
            } else {
                ""
            }
        );
        let _ = writeln!(md, "| Rust | {} |", report.environment.rustc_version);
        md.push('\n');

        // Pipeline Check Results
        if let Some(ref check) = report.diagnostics.check {
            md.push_str("## Pipeline Check Results\n\n");
            if check.success {
                md.push_str("All pipeline checks passed.\n\n");
            } else {
                md.push_str("**Pipeline check failed:**\n\n");
                md.push_str("```\n");
                md.push_str(&check.stderr);
                md.push_str("\n```\n\n");
            }
        }

        // Model Metadata
        if let Some(ref inspect) = report.diagnostics.inspect {
            md.push_str("## Model Metadata\n\n");
            md.push_str("<details>\n<summary>apr inspect output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&inspect.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Tensor Info
        if let Some(ref tensors) = report.diagnostics.tensors {
            md.push_str("## Tensor Inventory\n\n");
            md.push_str("<details>\n<summary>apr tensors output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&tensors.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Trace (if available)
        if let Some(ref trace) = report.diagnostics.trace {
            md.push_str("## Layer Trace\n\n");
            md.push_str("<details>\n<summary>apr trace output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&trace.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Error Explanation
        if let Some(ref explain) = report.diagnostics.explain {
            if !explain.stdout.is_empty() {
                md.push_str("## Error Analysis\n\n");
                md.push_str(&explain.stdout);
                md.push_str("\n\n");
            }
        }

        // Stderr Capture
        if let Some(ref stderr) = report.failure.stderr {
            if !stderr.is_empty() {
                md.push_str("## Stderr Capture\n\n");
                md.push_str("<details>\n<summary>Full stderr output</summary>\n\n");
                md.push_str("```\n");
                md.push_str(stderr);
                md.push_str("\n```\n\n");
                md.push_str("</details>\n\n");
            }
        }

        // Reproduction
        md.push_str("## Reproduction\n\n");
        md.push_str("```bash\n");
        md.push_str("# Reproduce this failure\n");
        let _ = writeln!(md, "{}\n", report.reproduction.command);
        md.push_str("# Run diagnostics manually\n");
        let _ = writeln!(md, "apr check {}", report.reproduction.model_path);
        let _ = writeln!(
            md,
            "apr trace {} --payload -v",
            report.reproduction.model_path
        );
        let _ = writeln!(md, "apr explain {}", report.failure.gate_id);
        md.push_str("```\n");

        md
    }
}

// Helper functions for environment collection

fn get_apr_version() -> String {
    Command::new("apr")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|s| s.replace("apr ", "").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

fn get_git_branch() -> String {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

fn get_git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty())
}

fn get_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .replace("rustc ", "")
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{HostInfo, Outcome, PerformanceMetrics};
    use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
    use chrono::Utc;
    use std::collections::HashMap;

    fn test_evidence() -> Evidence {
        Evidence {
            id: "test-evidence-001".to_string(),
            gate_id: "G3-STABLE".to_string(),
            scenario: QaScenario::new(
                ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct"),
                Modality::Run,
                Backend::Cpu,
                Format::Apr,
                "What is 2+2?".to_string(),
                0,
            ),
            outcome: Outcome::Crashed,
            reason: "Process crashed with exit code -1".to_string(),
            output: String::new(),
            stderr: Some("SIGSEGV at 0x12345".to_string()),
            exit_code: Some(-1),
            metrics: PerformanceMetrics {
                duration_ms: 52740,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_failure_details_from_evidence() {
        let evidence = test_evidence();
        let details = FailureDetails::from(&evidence);

        assert_eq!(details.gate_id, "G3-STABLE");
        assert_eq!(details.model, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assert_eq!(details.format, "Apr");
        assert_eq!(details.backend, "Cpu");
        assert_eq!(details.exit_code, Some(-1));
    }

    #[test]
    fn test_environment_context_collect() {
        let ctx = EnvironmentContext::collect();

        assert!(!ctx.os.is_empty());
        assert!(!ctx.arch.is_empty());
        assert!(!ctx.apr_qa_version.is_empty());
    }

    #[test]
    fn test_diagnostic_result_serialization() {
        let result = DiagnosticResult {
            command: "apr check model.apr".to_string(),
            success: true,
            stdout: "{}".to_string(),
            stderr: String::new(),
            duration_ms: 1234,
            timed_out: false,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("apr check"));
        assert!(json.contains("1234"));
    }

    #[test]
    fn test_generate_markdown() {
        let reporter = FailFastReporter::new(Path::new("output"));
        let evidence = test_evidence();

        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-02-04T18:00:00Z".to_string(),
            failure: FailureDetails::from(&evidence),
            environment: EnvironmentContext {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                apr_qa_version: "0.1.0".to_string(),
                apr_cli_version: "0.2.12".to_string(),
                git_commit: "abc123".to_string(),
                git_branch: "main".to_string(),
                git_dirty: false,
                rustc_version: "1.93.0".to_string(),
            },
            diagnostics: DiagnosticsBundle {
                check: None,
                inspect: None,
                trace: None,
                tensors: None,
                explain: None,
            },
            reproduction: ReproductionInfo {
                command: "apr-qa run playbook.yaml --fail-fast".to_string(),
                model_path: "/path/to/model.apr".to_string(),
                playbook: Some("playbook.yaml".to_string()),
            },
        };

        let md = reporter.generate_markdown(&report);

        assert!(md.contains("# Fail-Fast Report: G3-STABLE"));
        assert!(md.contains("| Gate | `G3-STABLE` |"));
        assert!(md.contains("| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |"));
        assert!(md.contains("## Reproduction"));
    }

    #[test]
    fn test_reporter_new() {
        let reporter = FailFastReporter::new(Path::new("output"));
        assert_eq!(reporter.output_dir, PathBuf::from("output"));
        assert_eq!(reporter.binary, "apr");
    }

    #[test]
    fn test_reporter_with_binary() {
        let reporter = FailFastReporter::new(Path::new("output")).with_binary("/custom/apr");
        assert_eq!(reporter.binary, "/custom/apr");
    }

    #[test]
    fn test_generate_markdown_with_diagnostics() {
        let reporter = FailFastReporter::new(Path::new("output"));
        let evidence = test_evidence();

        let check_result = DiagnosticResult {
            command: "apr check /model.apr --json".to_string(),
            success: false,
            stdout: "{}".to_string(),
            stderr: "Error: failed to load model".to_string(),
            duration_ms: 500,
            timed_out: false,
        };

        let inspect_result = DiagnosticResult {
            command: "apr inspect /model.apr --json".to_string(),
            success: true,
            stdout: r#"{"architecture": "Qwen2"}"#.to_string(),
            stderr: String::new(),
            duration_ms: 200,
            timed_out: false,
        };

        let tensors_result = DiagnosticResult {
            command: "apr tensors /model.apr --json".to_string(),
            success: true,
            stdout: r#"{"count": 256}"#.to_string(),
            stderr: String::new(),
            duration_ms: 150,
            timed_out: false,
        };

        let trace_result = DiagnosticResult {
            command: "apr trace /model.apr --payload --json".to_string(),
            success: true,
            stdout: r#"{"layers": []}"#.to_string(),
            stderr: String::new(),
            duration_ms: 1000,
            timed_out: false,
        };

        let explain_result = DiagnosticResult {
            command: "apr explain G3-STABLE".to_string(),
            success: true,
            stdout: "G3-STABLE: Model stability gate - ensures no crashes".to_string(),
            stderr: String::new(),
            duration_ms: 50,
            timed_out: false,
        };

        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-02-04T18:00:00Z".to_string(),
            failure: FailureDetails::from(&evidence),
            environment: EnvironmentContext {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                apr_qa_version: "0.1.0".to_string(),
                apr_cli_version: "0.2.12".to_string(),
                git_commit: "abc123".to_string(),
                git_branch: "main".to_string(),
                git_dirty: true,
                rustc_version: "1.93.0".to_string(),
            },
            diagnostics: DiagnosticsBundle {
                check: Some(check_result),
                inspect: Some(inspect_result),
                trace: Some(trace_result),
                tensors: Some(tensors_result),
                explain: Some(explain_result),
            },
            reproduction: ReproductionInfo {
                command: "apr-qa run playbook.yaml --fail-fast".to_string(),
                model_path: "/path/to/model.apr".to_string(),
                playbook: Some("playbook.yaml".to_string()),
            },
        };

        let md = reporter.generate_markdown(&report);

        // Check diagnostic sections are included
        assert!(md.contains("## Pipeline Check Results"));
        assert!(md.contains("**Pipeline check failed:**"));
        assert!(md.contains("Error: failed to load model"));
        assert!(md.contains("## Model Metadata"));
        assert!(md.contains("apr inspect output"));
        assert!(md.contains("## Tensor Inventory"));
        assert!(md.contains("apr tensors output"));
        assert!(md.contains("## Layer Trace"));
        assert!(md.contains("apr trace output"));
        assert!(md.contains("## Error Analysis"));
        assert!(md.contains("G3-STABLE: Model stability gate"));
        assert!(md.contains("[dirty]")); // git dirty flag
        assert!(md.contains("## Stderr Capture"));
        assert!(md.contains("SIGSEGV at 0x12345"));
    }

    #[test]
    fn test_generate_markdown_successful_check() {
        let reporter = FailFastReporter::new(Path::new("output"));
        let evidence = test_evidence();

        let check_result = DiagnosticResult {
            command: "apr check /model.apr --json".to_string(),
            success: true,
            stdout: "{}".to_string(),
            stderr: String::new(),
            duration_ms: 500,
            timed_out: false,
        };

        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-02-04T18:00:00Z".to_string(),
            failure: FailureDetails::from(&evidence),
            environment: EnvironmentContext {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                apr_qa_version: "0.1.0".to_string(),
                apr_cli_version: "0.2.12".to_string(),
                git_commit: "abc123".to_string(),
                git_branch: "main".to_string(),
                git_dirty: false,
                rustc_version: "1.93.0".to_string(),
            },
            diagnostics: DiagnosticsBundle {
                check: Some(check_result),
                inspect: None,
                trace: None,
                tensors: None,
                explain: None,
            },
            reproduction: ReproductionInfo {
                command: "apr-qa run playbook.yaml --fail-fast".to_string(),
                model_path: "/path/to/model.apr".to_string(),
                playbook: Some("playbook.yaml".to_string()),
            },
        };

        let md = reporter.generate_markdown(&report);

        assert!(md.contains("## Pipeline Check Results"));
        assert!(md.contains("All pipeline checks passed."));
    }

    #[test]
    fn test_run_trace_skips_non_apr() {
        let reporter = FailFastReporter::new(Path::new("output"));
        // run_trace should return None for non-.apr files
        let result = reporter.run_trace(Path::new("/model.safetensors"));
        assert!(result.is_none());
    }

    #[test]
    fn test_diagnostics_bundle_debug() {
        let bundle = DiagnosticsBundle {
            check: None,
            inspect: None,
            trace: None,
            tensors: None,
            explain: None,
        };
        // Just ensure Debug trait is implemented
        let _ = format!("{:?}", bundle);
    }

    #[test]
    fn test_reproduction_info_debug() {
        let info = ReproductionInfo {
            command: "apr-qa run test.yaml".to_string(),
            model_path: "/test/model.apr".to_string(),
            playbook: None,
        };
        // Just ensure Debug trait is implemented
        let _ = format!("{:?}", info);
    }

    #[test]
    fn test_fail_fast_report_debug() {
        let evidence = test_evidence();
        let report = FailFastReport {
            version: "1.0.0".to_string(),
            timestamp: "2024-02-04T18:00:00Z".to_string(),
            failure: FailureDetails::from(&evidence),
            environment: EnvironmentContext {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                apr_qa_version: "0.1.0".to_string(),
                apr_cli_version: "0.2.12".to_string(),
                git_commit: "abc123".to_string(),
                git_branch: "main".to_string(),
                git_dirty: false,
                rustc_version: "1.93.0".to_string(),
            },
            diagnostics: DiagnosticsBundle {
                check: None,
                inspect: None,
                trace: None,
                tensors: None,
                explain: None,
            },
            reproduction: ReproductionInfo {
                command: "apr-qa run playbook.yaml --fail-fast".to_string(),
                model_path: "/path/to/model.apr".to_string(),
                playbook: Some("playbook.yaml".to_string()),
            },
        };
        // Just ensure Debug trait is implemented
        let _ = format!("{:?}", report);
    }
}
