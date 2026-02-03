//! Fail-Fast Mode Demo
//!
//! Demonstrates the --fail-fast feature for debugging and GitHub ticket creation.
//!
//! Run with:
//! ```bash
//! cargo run --example fail_fast_demo
//! ```

use apr_qa_runner::{ExecutionConfig, FailurePolicy};

fn main() {
    println!("=== Fail-Fast Mode Demo ===\n");

    // Demonstrate FailurePolicy::FailFast
    let policy = FailurePolicy::FailFast;

    println!("FailurePolicy::FailFast properties:");
    println!("  emit_diagnostic(): {}", policy.emit_diagnostic());
    println!(
        "  stops_on_any_failure(): {}",
        policy.stops_on_any_failure()
    );

    // Show how to configure ExecutionConfig with FailFast
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::FailFast,
        ..Default::default()
    };

    println!("\nExecutionConfig with FailFast:");
    println!("  failure_policy: {:?}", config.failure_policy);
    println!("  default_timeout_ms: {}", config.default_timeout_ms);

    // Compare with other policies
    println!("\nPolicy comparison:");
    for (name, policy) in [
        ("StopOnFirst", FailurePolicy::StopOnFirst),
        ("StopOnP0", FailurePolicy::StopOnP0),
        ("CollectAll", FailurePolicy::CollectAll),
        ("FailFast", FailurePolicy::FailFast),
    ] {
        println!(
            "  {:<12} emit_diagnostic={:<5} stops_on_any={:<5}",
            name,
            policy.emit_diagnostic(),
            policy.stops_on_any_failure()
        );
    }

    println!("\n=== CLI Usage ===\n");
    println!("# Stop on first failure with enhanced diagnostics:");
    println!("apr-qa run playbook.yaml --fail-fast\n");

    println!("# Or equivalently:");
    println!("apr-qa run playbook.yaml --failure-policy fail-fast\n");

    println!("# With full tracing for GitHub ticket:");
    println!("RUST_LOG=debug apr-qa run playbook.yaml --fail-fast 2>&1 | tee failure.log\n");

    println!("=== Demo Complete ===");
}
