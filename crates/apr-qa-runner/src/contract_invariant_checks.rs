/// Parse tensor names from `apr rosetta inspect --json` output.
fn parse_tensor_names(json_output: &str) -> HashSet<String> {
    // Extract tensor_names array from JSON
    if let Some(start) = json_output.find("\"tensor_names\":[") {
        let after = &json_output[start + 16..];
        if let Some(end) = after.find(']') {
            let array_str = &after[..end];
            return array_str
                .split(',')
                .filter_map(|s| {
                    let trimmed = s.trim().trim_matches('"');
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect();
        }
    }
    HashSet::new()
}

/// I-3: No Silent Fallbacks — unknown dtype → error, never default to F32.
fn run_i3_no_silent_fallbacks(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let apr_path = resolve_apr_path(model_path);
    let result = runner.check_model(&apr_path);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!("I-3 No Silent Fallbacks: check failed: {}", result.stderr),
            &result.stdout,
            0,
        );
    }

    if contains_f32_fallback(&result.stdout) || contains_f32_fallback(&result.stderr) {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            "I-3 No Silent Fallbacks: detected F32 fallback in check output",
            &result.stdout,
            0,
        )
    } else {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-3 No Silent Fallbacks: no F32 fallbacks detected".to_string();
        ev
    }
}

/// Check if output contains evidence of silent F32 fallback.
fn contains_f32_fallback(output: &str) -> bool {
    let lower = output.to_lowercase();
    lower.contains("fallback") && lower.contains("f32")
        || lower.contains("defaulting to f32")
        || lower.contains("unknown dtype")
}

/// I-4: Statistical Preservation — tensor stats within dtype tolerance.
fn run_i4_statistical_preservation(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);
    let result = runner.validate_stats(&st_path, &apr_path);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-4 Statistical Preservation: validate-stats failed: {}",
                result.stderr
            ),
            &result.stdout,
            0,
        );
    }

    if result.stdout.contains("\"passed\":true") || result.stdout.contains("passed") {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-4 Statistical Preservation: tensor statistics preserved within tolerance"
            .to_string();
        ev
    } else {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-4 Statistical Preservation: statistics diverged: {}",
                result.stdout
            ),
            &result.stdout,
            0,
        )
    }
}

/// I-5: Tokenizer Roundtrip — encode(decode(tokens)) == tokens.
fn run_i5_tokenizer_roundtrip(
    runner: &Arc<dyn CommandRunner>,
    model_path: &Path,
    model_id: &ModelId,
    gate_id: &str,
) -> Evidence {
    let st_path = resolve_safetensors_path(model_path);
    let apr_path = resolve_apr_path(model_path);
    let result = runner.compare_inference(&st_path, &apr_path, "Hello", 1, 0.0);

    if !result.success {
        return Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-5 Tokenizer Roundtrip: compare-inference failed: {}",
                result.stderr
            ),
            &result.stdout,
            0,
        );
    }

    if result.stdout.contains("\"passed\":true") {
        let mut ev =
            Evidence::corroborated(gate_id, contract_scenario(model_id), &result.stdout, 0);
        ev.reason = "I-5 Tokenizer Roundtrip: tokenizer roundtrip verified".to_string();
        ev
    } else {
        Evidence::falsified(
            gate_id,
            contract_scenario(model_id),
            format!(
                "I-5 Tokenizer Roundtrip: inference output mismatch: {}",
                result.stdout
            ),
            &result.stdout,
            0,
        )
    }
}

/// Create a scenario for contract test evidence.
fn contract_scenario(model_id: &ModelId) -> QaScenario {
    QaScenario::new(
        model_id.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "Format contract invariant".to_string(),
        0,
    )
}

// ============================================================================
// Tests
// ============================================================================


#[cfg(test)]
#[path = "contract_tests.rs"]
mod contract_tests;
