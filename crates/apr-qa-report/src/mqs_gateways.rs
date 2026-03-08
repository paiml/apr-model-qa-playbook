impl MqsCalculator {
    /// Check gateway conditions (G0-G4)
    fn check_gateways(&self, evidence: &[Evidence]) -> Vec<GatewayResult> {
        let mut results = Vec::new();

        // G0: Model integrity (config/tensor consistency)
        // Checks for G0-INTEGRITY-* gate IDs
        let integrity_failures: Vec<&Evidence> = evidence
            .iter()
            .filter(|e| e.gate_id.starts_with("G0-INTEGRITY") && e.outcome.is_fail())
            .collect();
        if integrity_failures.is_empty() {
            results.push(GatewayResult::passed(
                "G0",
                "Model integrity (config/tensor match)",
            ));
        } else {
            let error_details: Vec<&str> = integrity_failures
                .iter()
                .map(|e| e.reason.as_str())
                .collect();
            results.push(GatewayResult::failed(
                "G0",
                "Model integrity (config/tensor match)",
                format!(
                    "{} integrity check(s) failed: {}",
                    integrity_failures.len(),
                    error_details.join("; ")
                ),
            ));
        }

        // G1: Model loads successfully
        let has_load_failure = evidence
            .iter()
            .any(|e| e.gate_id.starts_with("G1") && e.outcome.is_fail());
        if has_load_failure {
            results.push(GatewayResult::failed(
                "G1",
                "Model loads successfully",
                "Model failed to load",
            ));
        } else {
            results.push(GatewayResult::passed("G1", "Model loads successfully"));
        }

        // G2: Basic inference works
        let has_inference_failure = evidence
            .iter()
            .any(|e| e.gate_id.starts_with("G2") && e.outcome.is_fail());
        if has_inference_failure {
            results.push(GatewayResult::failed(
                "G2",
                "Basic inference works",
                "Inference failed",
            ));
        } else {
            results.push(GatewayResult::passed("G2", "Basic inference works"));
        }

        // G3: No crashes
        let crash_count = evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Crashed)
            .count();
        if crash_count > 0 {
            results.push(GatewayResult::failed(
                "G3",
                "No crashes",
                format!("{crash_count} crash(es) detected"),
            ));
        } else {
            results.push(GatewayResult::passed("G3", "No crashes"));
        }

        // G4: Output is not garbage
        let garbage_failures = evidence
            .iter()
            .filter(|e| e.gate_id.starts_with("G4") && e.outcome.is_fail())
            .count();
        if garbage_failures > evidence.len() / 4 {
            // More than 25% garbage output
            results.push(GatewayResult::failed(
                "G4",
                "Output is not garbage",
                format!("{garbage_failures} garbage outputs detected"),
            ));
        } else {
            results.push(GatewayResult::passed("G4", "Output is not garbage"));
        }

        results
    }

    /// Calculate category scores from evidence
    fn calculate_categories(&self, evidence: &[Evidence]) -> CategoryScores {
        // Tally pass/total per category using a map
        let mut tallies: HashMap<String, (usize, usize)> = HashMap::new();
        for e in evidence {
            let cat = Self::extract_category(&e.gate_id);
            let key = match cat.as_str() {
                "QUAL" | "PERF" | "STAB" | "COMP" | "EDGE" | "REGR" => cat,
                _ => "QUAL".to_string(), // Default unknown to QUAL
            };
            let entry = tallies.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if e.outcome.is_pass() {
                entry.0 += 1;
            }
        }

        let score_for = |cat: &str, max: u32| -> u32 {
            let &(pass, total) = tallies.get(cat).unwrap_or(&(0, 0));
            Self::proportional_score_or_full(pass, total, max)
        };

        // Categories with 0 tests score zero (Popper: untested ≠ qualified)
        CategoryScores {
            qual: score_for("QUAL", CategoryScores::MAX_QUAL),
            perf: score_for("PERF", CategoryScores::MAX_PERF),
            stab: score_for("STAB", CategoryScores::MAX_STAB),
            comp: score_for("COMP", CategoryScores::MAX_COMP),
            edge: score_for("EDGE", CategoryScores::MAX_EDGE),
            regr: score_for("REGR", CategoryScores::MAX_REGR),
        }
    }

    /// Extract MQS category from gate ID.
    ///
    /// Maps gate IDs to the 6 MQS categories via prefix lookup table,
    /// then falls back to parsing `F-{CATEGORY}-xxx` patterns.
    fn extract_category(gate_id: &str) -> String {
        // Prefix -> category mapping (order matters: longer prefixes first)
        const PREFIX_MAP: &[(&str, &str)] = &[
            ("F-CONV-RT", "REGR"),
            ("F-CONV-IDEM", "REGR"),
            ("F-CONV-COM", "REGR"),
            ("F-CONV", "COMP"),
            ("F-CONTRACT", "COMP"),
            ("G0-", "STAB"),
            ("T1-QUANT", "COMP"),   // Quantize → compatibility
            ("T2-IMPORT", "COMP"),  // Import → compatibility
            ("T3-PRUNE", "QUAL"),   // Prune → quality
            ("T4-DISTILL", "QUAL"), // Distill → quality
        ];

        for &(prefix, cat) in PREFIX_MAP {
            if gate_id.starts_with(prefix) {
                return cat.to_string();
            }
        }

        // Parse F-{CATEGORY}-xxx pattern
        const CATEGORIES: [&str; 6] = ["QUAL", "PERF", "STAB", "COMP", "EDGE", "REGR"];
        gate_id
            .split('-')
            .nth(1)
            .map(str::to_uppercase)
            .filter(|s| CATEGORIES.contains(&s.as_str()))
            .unwrap_or_else(|| "QUAL".to_string())
    }

    /// Calculate proportional score. Untested categories score zero.
    /// Rationale (Popper): absence of evidence means the category was never
    /// subjected to falsification — it cannot earn points it never defended.
    fn proportional_score_or_full(passed: usize, total: usize, max: u32) -> u32 {
        if total == 0 {
            return 0;
        }
        let ratio = passed as f64 / total as f64;
        (ratio * f64::from(max)).round() as u32
    }

    /// Normalize raw score to 0-100 using logarithmic scaling
    /// This makes achieving 100/100 extremely difficult
    #[cfg(test)]
    fn normalize_score(&self, raw: u32, pre_penalty: u32) -> f64 {
        self.normalize_score_with_max(raw, pre_penalty, CategoryScores::MAX_TOTAL)
    }

    /// Normalize with explicit max possible score.
    /// When proof bonus is active, max includes the bonus cap.
    fn normalize_score_with_max(&self, raw: u32, pre_penalty: u32, max_possible: u32) -> f64 {
        if pre_penalty == 0 {
            return 0.0;
        }

        let ratio = f64::from(raw) / f64::from(max_possible);

        // Apply logarithmic scaling to make high scores harder
        // f(x) = 100 * (log(1 + 9x) / log(10))
        // This maps [0,1] to [0,100] with diminishing returns
        let normalized = 100.0 * (1.0 + 9.0 * ratio).ln() / 10_f64.ln();

        // Clamp to valid range
        normalized.clamp(0.0, 100.0)
    }

    /// Calculate letter grade from normalized score
    fn calculate_grade(score: f64) -> String {
        const GRADE_TABLE: &[(f64, &str)] = &[
            (97.0, "A+"),
            (93.0, "A"),
            (90.0, "A-"),
            (87.0, "B+"),
            (83.0, "B"),
            (80.0, "B-"),
            (77.0, "C+"),
            (73.0, "C"),
            (70.0, "C-"),
            (67.0, "D+"),
            (63.0, "D"),
            (60.0, "D-"),
        ];
        GRADE_TABLE
            .iter()
            .find(|(threshold, _)| score >= *threshold)
            .map_or_else(|| "F".to_string(), |(_, grade)| (*grade).to_string())
    }
}
