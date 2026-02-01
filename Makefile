# APR Model QA Playbook
# Toyota Way + Popperian Falsification Testing Framework

.PHONY: all build test lint coverage clean check fmt doc \
       update-certifications certify-smoke certify-mvp certify-quick certify-standard certify-deep certify-qwen

# Default target
all: check

# Build all crates
build:
	cargo build --workspace

# Run all tests
test:
	cargo test --workspace

# Run tests with verbose output
test-verbose:
	cargo test --workspace -- --nocapture

# Run clippy lints
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Format code
fmt:
	cargo fmt --all

# Check formatting
fmt-check:
	cargo fmt --all -- --check

# Generate documentation
doc:
	cargo doc --workspace --no-deps

# Open documentation in browser
doc-open:
	cargo doc --workspace --no-deps --open

# Run all checks (build + lint + test)
check: fmt-check lint test

# Coverage with llvm-cov (NEVER use tarpaulin)
# Uses --lib to test library code only (no binary/main.rs)
coverage:
	cargo llvm-cov --workspace --lib --html
	@echo "Coverage report: target/llvm-cov/html/index.html"

# Coverage summary only (library code)
coverage-summary:
	cargo llvm-cov --workspace --lib

# Full coverage including CLI (for reference)
coverage-full:
	cargo llvm-cov --workspace --html
	@echo "Full coverage report: target/llvm-cov/html/index.html"

# Coverage with threshold check for PMAT compliance (95%)
# Uses --lib to exclude binary code (main.rs files)
# NO GAMING: All library code must meet the threshold
# Note: ~5% uncovered code is in subprocess execution paths requiring integration tests
# with actual apr binary - these are tested via E2E tests, not unit tests
coverage-check:
	@echo "Checking PMAT coverage compliance (>= 95%)..."
	@cargo llvm-cov --workspace --lib 2>&1 | \
		grep "^TOTAL" | awk '{print $$10}' | tr -d '%' | \
		xargs -I{} sh -c 'if [ $$(echo "{} >= 95" | bc) -eq 1 ]; then \
			echo "✓ Coverage: {}% (PMAT compliant)"; \
		else \
			echo "✗ Coverage: {}% (below 95% threshold)"; exit 1; \
		fi'

# Clean build artifacts
clean:
	cargo clean

# Run a specific playbook (example)
run-playbook:
	@echo "Usage: make run-playbook PLAYBOOK=examples/qwen-coder.yaml"
	@test -n "$(PLAYBOOK)" && cargo run --bin apr-qa -- run $(PLAYBOOK) || echo "Set PLAYBOOK variable"

# Generate MQS report
report:
	@echo "Usage: make report MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct"
	@test -n "$(MODEL)" && cargo run --bin apr-qa -- report $(MODEL) || echo "Set MODEL variable"

# Watch mode for development
watch:
	cargo watch -x "check --workspace" -x "test --workspace"

# Install development dependencies
dev-deps:
	cargo install cargo-watch cargo-llvm-cov

# Benchmark (if criterion tests exist)
bench:
	cargo bench --workspace

# Security audit
audit:
	cargo audit

# Update dependencies
update:
	cargo update

# Release build
release:
	cargo build --workspace --release

# ============================================================================
# Certification Targets (see docs/specifications/certified-testing.md)
# ============================================================================

# Update README certification table from CSV (REQUIRED after certification runs)
update-certifications:
	@echo "Updating README certification table from CSV..."
	cargo run --bin apr-qa-readme-sync --quiet

# Tiered certification targets
certify-smoke:
	@echo "Running Tier 1 (Smoke) certification - ~1-2 min per model..."
	cargo run --bin apr-qa -- certify --all --tier smoke

certify-mvp:
	@echo "Running Tier 2 (MVP) certification - all formats/backends/modalities (~5-10 min per model)..."
	cargo run --bin apr-qa -- certify --all --tier mvp

certify-quick:
	@echo "Running Tier 3 (Quick) certification - ~10-30 min per model..."
	cargo run --bin apr-qa -- certify --all --tier quick

certify-standard:
	@echo "Running Tier 3 (Standard) certification - 1 minute per model..."
	cargo run --bin apr-qa -- certify --all --tier standard

certify-deep:
	@echo "Running Tier 4 (Deep) certification - 10 minutes per model..."
	cargo run --bin apr-qa -- certify --all --tier deep

# Priority: Qwen Coder family (first qualification target)
certify-qwen:
	@echo "Running full certification for Qwen Coder family..."
	cargo run --bin apr-qa -- certify --family qwen-coder --tier deep
	@$(MAKE) update-certifications

# Help
help:
	@echo "APR Model QA Playbook - Make targets:"
	@echo ""
	@echo "  build                  Build all crates"
	@echo "  test                   Run all tests"
	@echo "  lint                   Run clippy lints"
	@echo "  fmt                    Format code"
	@echo "  check                  Run all checks (fmt, lint, test)"
	@echo "  coverage               Generate coverage report (library code)"
	@echo "  coverage-summary       Coverage summary (library code)"
	@echo "  coverage-full          Full coverage including CLI"
	@echo "  coverage-check         Verify PMAT compliance (>= 95%, see CLAUDE.md)"
	@echo "  doc                    Generate documentation"
	@echo "  clean                  Clean build artifacts"
	@echo "  watch                  Watch mode for development"
	@echo "  dev-deps               Install development dependencies"
	@echo ""
	@echo "Certification targets:"
	@echo "  update-certifications  Update README table from CSV"
	@echo "  certify-smoke          Tier 1: ~1-2 min per model (sanity check)"
	@echo "  certify-mvp            Tier 2: ~5-10 min per model (all formats/backends/modalities)"
	@echo "  certify-quick          Tier 3: ~10-30 min per model (dev iteration)"
	@echo "  certify-standard       Tier 4: ~1-2 hr per model (CI/CD)"
	@echo "  certify-deep           Tier 5: ~8-24 hr per model (production)"
	@echo "  certify-qwen           Full certification for Qwen Coder (priority)"
	@echo ""
	@echo "  help                   Show this help"
