# APR Model QA Playbook
# Toyota Way + Popperian Falsification Testing Framework

.PHONY: all build test lint coverage clean check fmt doc \
       update-certifications certify-smoke certify-mvp certify-quick certify-standard certify-deep certify-qwen \
       parity-check parity-list golden-generate \
       ci-smoke nightly-7b

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

# ============================================================================
# CI / Nightly Tier Targets (GH-6/AC-5)
# ============================================================================

# CI smoke: fastest possible qualification (1.5B, safetensors, CPU)
ci-smoke:
	@echo "Running CI smoke test (Qwen2.5-Coder-1.5B, ~1-2 min)..."
	cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-1.5b-smoke.playbook.yaml

# Nightly 7B: full qualification of primary QA model
nightly-7b:
	@echo "Running nightly 7B qualification (Qwen2.5-Coder-7B, ~30-60 min)..."
	cargo run --bin apr-qa -- run playbooks/models/qwen2.5-coder-7b-mvp.playbook.yaml

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
	@echo "CI / Nightly targets (GH-6/AC-5):"
	@echo "  ci-smoke               CI smoke: 1.5B safetensors CPU (~1-2 min)"
	@echo "  nightly-7b             Nightly: 7B MVP qualification (~30-60 min)"
	@echo ""
	@echo "HF Parity Oracle targets (cross-implementation validation):"
	@echo "  parity-check           Self-check all golden corpora"
	@echo "  parity-list            List available golden outputs"
	@echo "  golden-generate        Generate golden outputs (requires GPU + HuggingFace)"
	@echo ""
	@echo "  help                   Show this help"

# ============================================================================
# HF Parity Oracle Targets (see docs/specifications/hf-parity-oracle.md)
# ============================================================================

# Default golden corpus path
HF_CORPUS ?= ../hf-ground-truth-corpus/oracle

# Self-check all available model families in the golden corpus
# Implements Popperian self-consistency: golden outputs must match themselves
# Corpus structure: $(HF_CORPUS)/<model>/<version>/manifest.json
parity-check:
	@echo "=== HF Parity Self-Check (all model families) ==="
	@found=0; \
	for manifest in $$(find $(HF_CORPUS) -name "manifest.json" -type f 2>/dev/null); do \
		version_dir=$$(dirname $$manifest); \
		model_version=$$(echo $$version_dir | sed 's|$(HF_CORPUS)/||'); \
		echo "Checking: $$model_version"; \
		cargo run --bin apr-qa -- parity -m $$model_version -c $(HF_CORPUS) --self-check || exit 1; \
		echo ""; \
		found=1; \
	done; \
	if [ $$found -eq 0 ]; then echo "No golden corpora found in $(HF_CORPUS)"; exit 1; fi
	@echo "All parity self-checks passed!"

# List available golden outputs for all model families
# Corpus structure: $(HF_CORPUS)/<model>/<version>/manifest.json
parity-list:
	@echo "=== Available Golden Outputs ==="
	@for manifest in $$(find $(HF_CORPUS) -name "manifest.json" -type f 2>/dev/null); do \
		version_dir=$$(dirname $$manifest); \
		model_version=$$(echo $$version_dir | sed 's|$(HF_CORPUS)/||'); \
		echo ""; \
		echo "--- $$model_version ---"; \
		cargo run --bin apr-qa -- parity -m $$model_version -c $(HF_CORPUS) --list 2>/dev/null || true; \
	done

# Generate golden outputs using HuggingFace transformers
# Requires: uv, GPU with CUDA, HuggingFace model access
# Usage: make golden-generate MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct VERSION=v1
golden-generate:
	@echo "=== Generating Golden Outputs ==="
	@test -n "$(MODEL)" || (echo "Usage: make golden-generate MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct VERSION=v1"; exit 1)
	@test -n "$(VERSION)" || (echo "Usage: make golden-generate MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct VERSION=v1"; exit 1)
	@MODEL_SHORT=$$(echo $(MODEL) | sed 's|.*/||' | tr '[:upper:]' '[:lower:]'); \
	OUTPUT_DIR=$(HF_CORPUS)/$$MODEL_SHORT/$(VERSION); \
	echo "Model: $(MODEL)"; \
	echo "Output: $$OUTPUT_DIR"; \
	mkdir -p $$OUTPUT_DIR && \
	uv run scripts/generate_golden.py --model $(MODEL) --output $$OUTPUT_DIR --prompts prompts/code_bench.txt
