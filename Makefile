# APR Model QA Playbook
# Toyota Way + Popperian Falsification Testing Framework

.PHONY: all build test lint coverage clean check fmt doc

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
# Excludes CLI binary (hard to unit test) - library coverage must be >= 95%
coverage:
	cargo llvm-cov --workspace --html --ignore-filename-regex 'apr-qa-cli'
	@echo "Coverage report: target/llvm-cov/html/index.html"

# Coverage summary only (library code)
coverage-summary:
	cargo llvm-cov --workspace --ignore-filename-regex 'apr-qa-cli'

# Full coverage including CLI (for reference)
coverage-full:
	cargo llvm-cov --workspace --html
	@echo "Full coverage report: target/llvm-cov/html/index.html"

# Coverage with threshold check for PMAT compliance (95%)
coverage-check:
	@echo "Checking PMAT coverage compliance (>= 95%)..."
	@cargo llvm-cov --workspace --ignore-filename-regex 'apr-qa-cli' 2>&1 | \
		grep "^TOTAL" | awk '{print $$NF}' | tr -d '%' | \
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

# Help
help:
	@echo "APR Model QA Playbook - Make targets:"
	@echo ""
	@echo "  build            Build all crates"
	@echo "  test             Run all tests"
	@echo "  lint             Run clippy lints"
	@echo "  fmt              Format code"
	@echo "  check            Run all checks (fmt, lint, test)"
	@echo "  coverage         Generate coverage report (library code)"
	@echo "  coverage-summary Coverage summary (library code)"
	@echo "  coverage-full    Full coverage including CLI"
	@echo "  coverage-check   Verify PMAT compliance (>= 95%)"
	@echo "  doc              Generate documentation"
	@echo "  clean            Clean build artifacts"
	@echo "  watch            Watch mode for development"
	@echo "  dev-deps         Install development dependencies"
	@echo "  help             Show this help"
