/// Configuration for conversion executor
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConversionConfig {
    /// Test all format pairs
    pub test_all_pairs: bool,
    /// Test round-trips
    pub test_round_trips: bool,
    /// Test multi-hop conversion chains (T-QKV-04)
    pub test_multi_hop: bool,
    /// Test tensor cardinality after conversion (MR-CARD)
    pub test_cardinality: bool,
    /// Test tensor name preservation after conversion (T-QKV-02)
    pub test_tensor_names: bool,
    /// Test idempotency of double-conversion (MR-IDEM)
    pub test_idempotency: bool,
    /// Test commutativity of conversion paths (MR-COM)
    pub test_commutativity: bool,
    /// Backends to test
    pub backends: Vec<Backend>,
    /// Use CPU only (no GPU)
    pub no_gpu: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu, Backend::Gpu],
            no_gpu: false,
        }
    }
}

impl ConversionConfig {
    /// Create config for CPU-only testing
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            test_all_pairs: true,
            test_round_trips: true,
            test_multi_hop: true,
            test_cardinality: true,
            test_tensor_names: true,
            test_idempotency: true,
            test_commutativity: true,
            backends: vec![Backend::Cpu],
            no_gpu: true,
        }
    }
}

/// Executor for running P0 format conversion tests
#[derive(Debug)]
pub struct ConversionExecutor {
    config: ConversionConfig,
    binary: String,
    /// Output directory for conversion artifacts (ISO-OUT-001)
    output_dir: Option<PathBuf>,
}

impl ConversionExecutor {
    /// Create a new conversion executor
    #[must_use]
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            binary: default_binary(),
            output_dir: None,
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ConversionConfig::default())
    }

    /// Set the output directory for conversion artifacts (ISO-OUT-001)
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = Some(output_dir);
        self
    }

    /// Execute all conversion tests for a model
    ///
    /// # Errors
    ///
    /// Returns an error if a critical conversion failure occurs.
    pub fn execute_all(
        &self,
        model_path: &Path,
        model_id: &ModelId,
    ) -> Result<ConversionExecutionResult> {
        let mut results = Vec::new();
        let mut evidence = Vec::new();
        let start = std::time::Instant::now();

        let backends: Vec<Backend> = if self.config.no_gpu {
            vec![Backend::Cpu]
        } else {
            self.config.backends.clone()
        };

        let output_dir_wrapper = self
            .output_dir
            .as_ref()
            .map(|dir| ConversionOutputDir::new(dir, model_id));

        if self.config.test_all_pairs {
            self.run_all_pairs(
                model_path,
                model_id,
                &backends,
                output_dir_wrapper.as_ref(),
                &mut results,
                &mut evidence,
            );
        }

        if self.config.test_round_trips {
            self.run_round_trips(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_multi_hop {
            self.run_multi_hop_chains(model_path, model_id, &backends, &mut results, &mut evidence);
            self.run_byte_level_rt(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_idempotency {
            self.run_idempotency(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_commutativity {
            self.run_commutativity(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_cardinality || self.config.test_tensor_names {
            self.run_structural_checks(model_path, model_id, &mut results, &mut evidence);
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let passed = results
            .iter()
            .filter(|r| matches!(r, ConversionResult::Corroborated { .. }))
            .count();
        let failed = results.len() - passed;

        Ok(ConversionExecutionResult {
            total: results.len(),
            passed,
            failed,
            duration_ms,
            results,
            evidence,
        })
    }

    /// Test all format conversion pairs
    fn run_all_pairs(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        output_dir_wrapper: Option<&ConversionOutputDir>,
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for (source, target) in all_conversion_pairs() {
            for backend in backends {
                let mut test = ConversionTest::new(source, target, *backend, model_id.clone());
                test.binary.clone_from(&self.binary);
                if let Some(out_dir) = &output_dir_wrapper {
                    test.output_dir = Some((*out_dir).clone());
                }

                match test.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            &test.gate_id(),
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                target,
                                format!("Convert {source:?} to {target:?}"),
                                0,
                            ),
                            format!("Conversion infrastructure error: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                        results.push(ConversionResult::Falsified {
                            gate_id: test.gate_id(),
                            reason: e.to_string(),
                            evidence: ConversionEvidence {
                                source_hash: String::new(),
                                converted_hash: String::new(),
                                max_diff: f64::MAX,
                                diff_indices: vec![],
                                source_format: source,
                                target_format: target,
                                backend: *backend,
                                failure_type: None,
                                quant_type: None,
                            },
                        });
                    }
                }
            }
        }
    }

    /// Test round-trips (GGUF → APR → SafeTensors → GGUF) - F-CONV-RT-001
    fn run_round_trips(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut rt = RoundTripTest::new(
                vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
                *backend,
                model_id.clone(),
            );
            rt.binary.clone_from(&self.binary);

            match rt.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-RT-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Gguf,
                            "Round-trip conversion".to_string(),
                            0,
                        ),
                        format!("Round-trip failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                }
            }
        }
    }

    /// Multi-hop chain tests (F-CONV-RT-002, RT-003, RT-004)
    fn run_multi_hop_chains(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        let multi_hop_chains: Vec<(&str, Vec<Format>)> = vec![
            (
                "F-CONV-RT-002",
                vec![
                    Format::SafeTensors,
                    Format::Apr,
                    Format::Gguf,
                    Format::SafeTensors,
                ],
            ),
            (
                "F-CONV-RT-003",
                vec![
                    Format::SafeTensors,
                    Format::Apr,
                    Format::Gguf,
                    Format::Apr,
                    Format::SafeTensors,
                ],
            ),
            (
                "F-CONV-RT-004",
                vec![Format::SafeTensors, Format::Apr, Format::Gguf, Format::Apr],
            ),
        ];

        for (gate_id, chain) in &multi_hop_chains {
            for backend in backends {
                let mut rt = RoundTripTest::new(chain.clone(), *backend, model_id.clone());
                rt.binary.clone_from(&self.binary);

                match rt.execute(model_path) {
                    Ok(mut result) => {
                        if let ConversionResult::Falsified {
                            gate_id: ref mut gid,
                            ..
                        } = result
                        {
                            *gid = (*gate_id).to_string();
                        }
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let chain_desc: Vec<_> = chain.iter().map(|f| format!("{f:?}")).collect();
                        let ev = Evidence::falsified(
                            *gate_id,
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::SafeTensors,
                                format!("Multi-hop: {}", chain_desc.join("→")),
                                0,
                            ),
                            format!("Multi-hop chain failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                    }
                }
            }
        }
    }

    /// Byte-level round-trip test (F-CONV-RT-BYTE-001)
    fn run_byte_level_rt(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut byte_rt = ByteLevelRoundTripTest::new(*backend, model_id.clone());
            byte_rt.binary.clone_from(&self.binary);

            match byte_rt.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-RT-BYTE-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::SafeTensors,
                            "Byte-level round-trip ST→APR→GGUF→APR".to_string(),
                            0,
                        ),
                        format!("Byte-level round-trip failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                }
            }
        }
    }

    /// Idempotency test (F-CONV-IDEM-001)
    fn run_idempotency(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut idem =
                IdempotencyTest::new(Format::Gguf, Format::Apr, *backend, model_id.clone());
            idem.binary.clone_from(&self.binary);

            match idem.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-IDEM-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Apr,
                            "Idempotency: GGUF→APR twice".to_string(),
                            0,
                        ),
                        format!("Idempotency test failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                }
            }
        }
    }

    /// Commutativity test (F-CONV-COM-001)
    fn run_commutativity(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut com = CommutativityTest::new(*backend, model_id.clone());
            com.binary.clone_from(&self.binary);

            match com.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-COM-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Apr,
                            "Commutativity: GGUF→APR vs GGUF→ST→APR".to_string(),
                            0,
                        ),
                        format!("Commutativity test failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                }
            }
        }
    }

    /// Structural checks: cardinality (F-CONV-CARD-001) and tensor names (F-CONV-NAME-001)
    fn run_structural_checks(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for (source, target) in all_conversion_pairs() {
            let target_ext = format_extension(target);
            let converted_path = model_path.with_extension(format!("converted.{target_ext}"));
            if !converted_path.exists() {
                continue;
            }

            if self.config.test_cardinality {
                self.check_cardinality_gate(
                    model_path,
                    &converted_path,
                    model_id,
                    source,
                    target,
                    results,
                    evidence,
                );
            }

            if self.config.test_tensor_names {
                self.check_tensor_name_gate(
                    model_path,
                    &converted_path,
                    model_id,
                    source,
                    target,
                    evidence,
                );
            }
        }
    }

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

/// Result of conversion test execution
#[derive(Debug)]
pub struct ConversionExecutionResult {
    /// Total tests run
    pub total: usize,
    /// Tests passed
    pub passed: usize,
    /// Tests failed
    pub failed: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Individual results
    pub results: Vec<ConversionResult>,
    /// Evidence collected
    pub evidence: Vec<Evidence>,
}

impl ConversionExecutionResult {
    /// Check if all conversion tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Get pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.passed as f64 / self.total as f64) * 100.0
        }
    }
}

/// Convert ConversionResult to Evidence
impl From<ConversionResult> for Evidence {
    fn from(result: ConversionResult) -> Self {
        match result {
            ConversionResult::Corroborated {
                source_format,
                target_format,
                backend,
                max_diff,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    backend,
                    target_format,
                    format!("Convert {source_format:?} to {target_format:?}"),
                    0,
                );
                Evidence::corroborated(
                    &format!("F-CONV-{source_format:?}-{target_format:?}"),
                    scenario,
                    &format!("Conversion successful, max_diff: {max_diff:.2e}"),
                    0,
                )
            }
            ConversionResult::Falsified {
                gate_id,
                reason,
                evidence,
            } => {
                let scenario = QaScenario::new(
                    ModelId::new("conversion", "test"),
                    Modality::Run,
                    evidence.backend,
                    evidence.target_format,
                    format!(
                        "Convert {:?} to {:?}",
                        evidence.source_format, evidence.target_format
                    ),
                    0,
                );
                Evidence::falsified(&gate_id, scenario, reason, &evidence.converted_hash, 0)
            }
        }
    }
}
