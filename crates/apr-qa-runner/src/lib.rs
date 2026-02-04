//! APR QA Runner
//!
//! Playbook executor for model qualification testing.
//! Implements parallel execution with Jidoka (stop-on-failure) support.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Allow common patterns
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unused_self)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::single_char_pattern)]
// Allow common patterns in test code
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
#![cfg_attr(test, allow(clippy::uninlined_format_args))]
#![cfg_attr(test, allow(clippy::cast_sign_loss))]

pub mod command;
pub mod conversion;
pub mod differential;
pub mod error;
pub mod evidence;
pub mod executor;
pub mod integrity;
pub mod oracle;
pub mod parallel;
pub mod patterns;
pub mod playbook;
pub mod process;
pub mod provenance;
pub use provenance::{
    DerivedProvenance, Provenance, ProvenanceError, SourceProvenance, add_derived, compute_sha256,
    create_source_provenance, get_apr_cli_version, load_provenance, save_provenance,
    validate_comparison, validate_provenance, verify_files_exist, verify_provenance_integrity,
};

#[cfg(test)]
pub mod test_fixtures;

pub use command::{CommandOutput, CommandRunner, MockCommandRunner, RealCommandRunner};
pub use conversion::{
    CommutativityTest, ConversionBugType, ConversionConfig, ConversionEvidence,
    ConversionExecutionResult, ConversionExecutor, ConversionFailureType, ConversionResult,
    ConversionTest, ConversionTolerance, DEFAULT_TOLERANCES, EPSILON, IdempotencyTest, QuantType,
    RoundTripTest, SemanticConversionTest, SemanticTestResult, TensorNaming, all_backends,
    all_conversion_pairs, check_cardinality, check_tensor_names, classify_failure,
    generate_conversion_tests, get_hf_cache_dir, resolve_hf_repo_to_cache, resolve_model_path,
    split_hf_repo, tolerance_for,
};
pub use differential::{
    BenchResult, BenchmarkMetrics, CiAssertion, CiProfileResult, DiffBenchmarkResult, DiffConfig,
    DifferentialExecutor, FormatConversionResult, InferenceComparisonResult, InspectResult,
    ModelPreparationResult, ProfileAssertion, SixColumnProfile, TensorDiffResult, TensorMismatch,
    TensorMismatchType, TokenComparison, convert_format_cached, prepare_model_with_provenance,
    run_bench_throughput, run_diff_benchmark, run_inspect, run_profile_ci, run_six_column_profile,
    verify_comparison_provenance,
};
pub use error::{Error, Result};
pub use evidence::{Evidence, EvidenceCollector, Outcome, PerformanceMetrics};
pub use executor::{
    ExecutionConfig, ExecutionResult, Executor, FailurePolicy, ToolExecutor, ToolTestResult,
};
pub use integrity::{
    ConfigValues, IntegrityResult, TensorDerivedValues, check_safetensors_integrity,
    gate_ids as integrity_gate_ids,
};
pub use oracle::{
    CheckStatus, Confidence, CrossReference, FalsificationCheckItem, OracleContext, OracleEnhancer,
    OracleError, RankedHypothesis, generate_checklist_markdown,
};
pub use parallel::{ParallelConfig, ParallelExecutor, ParallelResult};
pub use patterns::{
    ApiComplianceChecker, ApiComplianceResult, BugPattern, CompanionCheckResult, DosCheckResult,
    DosProtectionConfig, DosViolation, IntegrityCheckResult, IntegrityChecker,
    NumericalStabilityResult, ParityCheckResult, ParityChecker, PathSafetyResult, PathViolation,
    PatternDetector, PerformanceCheckResult, PerformanceThresholds, PerformanceValidator,
    PromptPattern, PromptSafetyResult, SpecGate, TensorValidityResult,
};
pub use playbook::{
    DifferentialTestConfig, FingerprintConfig, FormatValidationConfig, InferenceCompareConfig,
    Playbook, PlaybookLockEntry, PlaybookLockFile, PlaybookStep, ProfileCiAssertions,
    ProfileCiConfig, SkipReason, SkipType, StatsToleranceConfig, TensorDiffConfig,
    TracePayloadConfig, ValidateStatsConfig, compute_playbook_hash, detect_implicit_skips,
    find_skip_files, generate_lock_entry, load_lock_file, save_lock_file,
    verify_playbook_integrity,
};
