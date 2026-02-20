impl ConversionExecutor {

    /// Check cardinality gate for a single conversion pair
    #[allow(clippy::too_many_arguments)]
    fn check_cardinality_gate(
        &self,
        model_path: &Path,
        converted_path: &Path,
        model_id: &ModelId,
        source: Format,
        target: Format,
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        match check_cardinality(model_path, converted_path, &self.binary) {
            Ok(Some((gate_id, reason))) => {
                let ev = Evidence::falsified(
                    &gate_id,
                    QaScenario::new(
                        model_id.clone(),
                        Modality::Run,
                        Backend::Cpu,
                        target,
                        format!("Cardinality {source:?}→{target:?}"),
                        0,
                    ),
                    &reason,
                    "N/A",
                    0,
                );
                evidence.push(ev);
                results.push(ConversionResult::Falsified {
                    gate_id: "F-CONV-CARD-001".to_string(),
                    reason,
                    evidence: ConversionEvidence {
                        source_hash: String::new(),
                        converted_hash: String::new(),
                        max_diff: 0.0,
                        diff_indices: vec![],
                        source_format: source,
                        target_format: target,
                        backend: Backend::Cpu,
                        failure_type: None,
                        quant_type: None,
                    },
                });
            }
            Ok(None) => {
                let ev = Evidence::corroborated(
                    "F-CONV-CARD-001",
                    QaScenario::new(
                        model_id.clone(),
                        Modality::Run,
                        Backend::Cpu,
                        target,
                        format!("Cardinality {source:?}→{target:?}"),
                        0,
                    ),
                    "Tensor cardinality preserved",
                    0,
                );
                evidence.push(ev);
            }
            Err(_) => {} // Inspect not available, skip gate
        }
    }

    /// Check tensor name preservation gate for a single conversion pair
    fn check_tensor_name_gate(
        &self,
        model_path: &Path,
        converted_path: &Path,
        model_id: &ModelId,
        source: Format,
        target: Format,
        evidence: &mut Vec<Evidence>,
    ) {
        match check_tensor_names(model_path, converted_path, &self.binary) {
            Ok(Some((gate_id, reason))) => {
                let ev = Evidence::falsified(
                    &gate_id,
                    QaScenario::new(
                        model_id.clone(),
                        Modality::Run,
                        Backend::Cpu,
                        target,
                        format!("Tensor names {source:?}→{target:?}"),
                        0,
                    ),
                    &reason,
                    "N/A",
                    0,
                );
                evidence.push(ev);
            }
            Ok(None) => {
                let ev = Evidence::corroborated(
                    "F-CONV-NAME-001",
                    QaScenario::new(
                        model_id.clone(),
                        Modality::Run,
                        Backend::Cpu,
                        target,
                        format!("Tensor names {source:?}→{target:?}"),
                        0,
                    ),
                    "Tensor names preserved",
                    0,
                );
                evidence.push(ev);
            }
            Err(_) => {} // Inspect not available, skip gate
        }
    }
}
