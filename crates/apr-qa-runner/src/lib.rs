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

pub mod conversion;
pub mod error;
pub mod evidence;
pub mod executor;
pub mod parallel;
pub mod playbook;

pub use conversion::{
    ConversionConfig, ConversionEvidence, ConversionExecutionResult, ConversionExecutor,
    ConversionResult, ConversionTest, EPSILON, RoundTripTest, all_backends, all_conversion_pairs,
    generate_conversion_tests,
};
pub use error::{Error, Result};
pub use evidence::{Evidence, EvidenceCollector, Outcome, PerformanceMetrics};
pub use executor::{ExecutionConfig, Executor, FailurePolicy, ToolExecutor, ToolTestResult};
pub use parallel::{ExecutionMode, ParallelConfig, ParallelExecutor, ParallelResult};
pub use playbook::{Playbook, PlaybookStep};
