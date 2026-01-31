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
pub mod parallel;
pub mod patterns;
pub mod playbook;
pub mod process;

pub use command::{CommandOutput, CommandRunner, MockCommandRunner, RealCommandRunner};
pub use conversion::{
    ConversionBugType, ConversionConfig, ConversionEvidence, ConversionExecutionResult,
    ConversionExecutor, ConversionResult, ConversionTest, EPSILON, RoundTripTest,
    SemanticConversionTest, SemanticTestResult, all_backends, all_conversion_pairs,
    generate_conversion_tests,
};
pub use differential::{
    BenchResult, BenchmarkMetrics, CiAssertion, CiProfileResult, DiffBenchmarkResult, DiffConfig,
    DifferentialExecutor, FormatConversionResult, InferenceComparisonResult, ProfileAssertion,
    SixColumnProfile, TensorDiffResult, TensorMismatch, TensorMismatchType, TokenComparison,
    convert_format_cached, run_bench_throughput, run_diff_benchmark, run_profile_ci,
    run_six_column_profile,
};
pub use error::{Error, Result};
pub use evidence::{Evidence, EvidenceCollector, Outcome, PerformanceMetrics};
pub use executor::{
    ExecutionConfig, ExecutionResult, Executor, FailurePolicy, ToolExecutor, ToolTestResult,
};
pub use parallel::{ExecutionMode, ParallelConfig, ParallelExecutor, ParallelResult};
pub use patterns::{
    ApiComplianceChecker, ApiComplianceResult, BugPattern, CompanionCheckResult, DosCheckResult,
    DosProtectionConfig, DosViolation, IntegrityCheckResult, IntegrityChecker,
    NumericalStabilityResult, ParityCheckResult, ParityChecker, PathSafetyResult, PathViolation,
    PatternDetector, PerformanceCheckResult, PerformanceThresholds, PerformanceValidator,
    PromptPattern, PromptSafetyResult, SpecGate, TensorValidityResult,
};
pub use playbook::{
    DifferentialTestConfig, FingerprintConfig, FormatValidationConfig, InferenceCompareConfig,
    Playbook, PlaybookStep, ProfileCiAssertions, ProfileCiConfig, StatsToleranceConfig,
    TensorDiffConfig, TracePayloadConfig, ValidateStatsConfig,
};
