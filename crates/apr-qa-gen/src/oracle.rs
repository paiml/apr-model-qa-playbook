//! Oracle definitions for output verification
//!
//! Oracles are pure functions that verify model output correctness.
//! Each oracle implements Popperian falsification - it attempts to
//! disprove the hypothesis that the model output is correct.
//!
//! # Design
//!
//! An oracle returns `OracleResult::Corroborated` when it fails to
//! disprove correctness, and `OracleResult::Falsified` when it
//! successfully disproves the hypothesis.

use serde::{Deserialize, Serialize};

/// Result of oracle evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OracleResult {
    /// Hypothesis not falsified - output appears correct
    Corroborated {
        /// Evidence supporting corroboration
        evidence: String,
    },
    /// Hypothesis falsified - output is incorrect
    Falsified {
        /// Reason for falsification
        reason: String,
        /// Evidence of failure
        evidence: String,
    },
}

impl OracleResult {
    /// Check if the result is corroborated
    #[must_use]
    pub const fn is_corroborated(&self) -> bool {
        matches!(self, Self::Corroborated { .. })
    }

    /// Check if the result is falsified
    #[must_use]
    pub const fn is_falsified(&self) -> bool {
        matches!(self, Self::Falsified { .. })
    }
}

/// Oracle trait for output verification
pub trait Oracle: Send + Sync {
    /// Evaluate the output against the prompt
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult;

    /// Get the oracle name
    fn name(&self) -> &'static str;
}

/// Arithmetic oracle - verifies mathematical correctness
#[derive(Debug, Clone, Default)]
pub struct ArithmeticOracle;

impl ArithmeticOracle {
    /// Create a new arithmetic oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Try to parse and evaluate a simple arithmetic expression
    fn eval_arithmetic(expr: &str) -> Option<i64> {
        // Simple parser for "a+b", "a-b", "a*b", "a/b"
        let expr = expr.trim().trim_end_matches('=').trim_end_matches('?');

        // Find the FIRST operator by string position, not by operator priority.
        // Searching by operator type [+,-,*,/] is wrong: "3-2+1" would find '+'
        // at position 3, then try to parse "3-2" as i64, which fails.
        // Skip position 0 for '-' to handle negative numbers like "-5+3".
        let first_op = ['+', '-', '*', '/']
            .iter()
            .filter_map(|&op| {
                expr.find(op)
                    .and_then(|pos| {
                        // Skip '-' at position 0 (negative sign, not subtraction)
                        if pos == 0 && op == '-' {
                            expr[1..].find(op).map(|p| (p + 1, op))
                        } else {
                            Some((pos, op))
                        }
                    })
            })
            .min_by_key(|&(pos, _)| pos);

        if let Some((pos, op)) = first_op {
            let left: i64 = expr[..pos].trim().parse().ok()?;
            let right: i64 = expr[pos + 1..].trim().parse().ok()?;
            return match op {
                '+' => Some(left + right),
                '-' => Some(left - right),
                '*' => Some(left * right),
                '/' if right != 0 => Some(left / right),
                _ => None,
            };
        }
        None
    }
}

impl Oracle for ArithmeticOracle {
    /// Evaluate arithmetic correctness by checking if output contains expected value
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        // Try to extract arithmetic expression from prompt
        let Some(expected) = Self::eval_arithmetic(prompt) else {
            // Not an arithmetic prompt, skip
            return OracleResult::Corroborated {
                evidence: "Non-arithmetic prompt, skipped".to_string(),
            };
        };

        // Check if output contains the expected value
        if output.contains(&expected.to_string()) {
            OracleResult::Corroborated {
                evidence: format!("Found expected value {expected} in output"),
            }
        } else {
            OracleResult::Falsified {
                reason: format!("Expected {expected} not found in output"),
                evidence: format!("Output: {}", truncate(output, 100)),
            }
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "arithmetic"
    }
}

/// Garbage detection oracle - verifies output is not garbage
#[derive(Debug, Clone, Default)]
pub struct GarbageOracle;

impl GarbageOracle {
    /// Create a new garbage oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Oracle for GarbageOracle {
    /// Check output for garbage patterns including empty, control chars, NaN, and repetition
    fn evaluate(&self, _prompt: &str, output: &str) -> OracleResult {
        // Check for empty output
        if output.trim().is_empty() {
            return OracleResult::Falsified {
                reason: "Output is empty".to_string(),
                evidence: "Empty output".to_string(),
            };
        }

        // Check for control characters (except newline, tab)
        let control_chars: Vec<char> = output
            .chars()
            .filter(|c| c.is_control() && *c != '\n' && *c != '\t' && *c != '\r')
            .collect();
        if !control_chars.is_empty() {
            return OracleResult::Falsified {
                reason: "Output contains control characters".to_string(),
                evidence: format!("Found {} control chars", control_chars.len()),
            };
        }

        // Check for NaN/Inf (numerical explosion)
        if output.contains("NaN") || output.contains("Inf") || output.contains("inf") {
            return OracleResult::Falsified {
                reason: "Output contains NaN or Inf".to_string(),
                evidence: format!("Output: {}", truncate(output, 100)),
            };
        }

        // Check for repetitive patterns (e.g., "akakakakak")
        if is_repetitive(output) {
            return OracleResult::Falsified {
                reason: "Output is highly repetitive".to_string(),
                evidence: format!("Output: {}", truncate(output, 100)),
            };
        }

        // Check for replacement character (encoding issues)
        if output.contains('\u{FFFD}') {
            return OracleResult::Falsified {
                reason: "Output contains replacement characters".to_string(),
                evidence: "Found U+FFFD replacement character".to_string(),
            };
        }

        OracleResult::Corroborated {
            evidence: format!("Valid output ({} chars)", output.len()),
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "garbage"
    }
}

/// Code syntax oracle - verifies output looks like code
#[derive(Debug, Clone, Default)]
pub struct CodeSyntaxOracle;

impl CodeSyntaxOracle {
    /// Create a new code syntax oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Oracle for CodeSyntaxOracle {
    /// Verify output contains code-like patterns after garbage check
    #[allow(clippy::used_underscore_binding)]
    fn evaluate(&self, _prompt: &str, output: &str) -> OracleResult {
        // First check for garbage
        let garbage_oracle = GarbageOracle::new();
        if let OracleResult::Falsified { reason, evidence } =
            garbage_oracle.evaluate(_prompt, output)
        {
            return OracleResult::Falsified { reason, evidence };
        }

        // Check for code-like patterns
        let code_indicators = [
            "fn ",
            "def ",
            "function ",
            "class ",
            "struct ",
            "impl ",
            "pub ",
            "let ",
            "const ",
            "var ",
            "if ",
            "for ",
            "while ",
            "return ",
            "import ",
            "from ",
            "use ",
            "{",
            "}",
            "(",
            ")",
            ";",
            "=>",
            "->",
        ];

        let has_code_pattern = code_indicators.iter().any(|p| output.contains(p));

        // Very short output might just be a completion of a function signature
        if has_code_pattern || output.len() < 20 {
            OracleResult::Corroborated {
                evidence: "Output appears to be valid code".to_string(),
            }
        } else {
            // Not necessarily a failure - might be a docstring or comment
            OracleResult::Corroborated {
                evidence: "Output may be code documentation".to_string(),
            }
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "code_syntax"
    }
}

/// Combined oracle that runs multiple oracles
pub struct CompositeOracle {
    /// Oracle display name
    name: &'static str,
    /// Child oracles evaluated in order
    oracles: Vec<Box<dyn Oracle + Send + Sync>>,
}

impl std::fmt::Debug for CompositeOracle {
    /// Format the composite oracle showing name and child count
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeOracle")
            .field("name", &self.name)
            .field("oracle_count", &self.oracles.len())
            .finish()
    }
}

// Manual Clone implementation since Box<dyn Oracle> doesn't implement Clone
impl CompositeOracle {
    /// Create a new composite oracle
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            oracles: Vec::new(),
        }
    }

    /// Add an oracle to the composite
    pub fn add<O: Oracle + Clone + 'static>(&mut self, oracle: O) {
        self.oracles.push(Box::new(oracle));
    }
}

// We need a wrapper to make the oracles cloneable
/// Wrapper to enable cloning of boxed oracle trait objects
#[allow(dead_code)]
struct OracleWrapper<O: Oracle + Clone>(O);

impl<O: Oracle + Clone> Oracle for OracleWrapper<O> {
    /// Delegate evaluation to the wrapped oracle
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        self.0.evaluate(prompt, output)
    }

    /// Return the wrapped oracle's name
    fn name(&self) -> &'static str {
        self.0.name()
    }
}

impl Oracle for CompositeOracle {
    /// Evaluate all child oracles, returning first falsification or overall corroboration
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        for oracle in &self.oracles {
            if let result @ OracleResult::Falsified { .. } = oracle.evaluate(prompt, output) {
                return result;
            }
        }
        OracleResult::Corroborated {
            evidence: format!("All {} oracles passed", self.oracles.len()),
        }
    }

    /// Return the composite oracle's name
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Select the appropriate oracle based on prompt characteristics
#[must_use]
pub fn select_oracle(prompt: &str) -> Box<dyn Oracle + Send + Sync> {
    if is_arithmetic_prompt(prompt) {
        Box::new(ArithmeticOracle::new())
    } else if is_code_prompt(prompt) {
        Box::new(CodeSyntaxOracle::new())
    } else {
        Box::new(GarbageOracle::new())
    }
}

/// Check if prompt is an arithmetic question
fn is_arithmetic_prompt(prompt: &str) -> bool {
    let prompt_lower = prompt.to_lowercase();
    (prompt_lower.contains('+')
        || prompt_lower.contains('-')
        || prompt_lower.contains('*')
        || prompt_lower.contains('/'))
        && prompt.chars().any(|c| c.is_ascii_digit())
}

/// Check if prompt is a code completion request
fn is_code_prompt(prompt: &str) -> bool {
    prompt.starts_with("def ")
        || prompt.starts_with("fn ")
        || prompt.starts_with("function ")
        || prompt.starts_with("class ")
        || prompt.starts_with("async ")
        || prompt.contains("```")
}

/// Check if a string contains a repeating substring pattern
///
/// For each candidate period `p` in `[2, min(20, len/3)]`, extracts the first
/// `p` bytes as a pattern and counts consecutive repetitions from the start.
/// Returns true if reps >= 3 AND coverage >= 70% of the string length.
fn check_substring_repetition(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len < 6 {
        return false;
    }
    let max_period = 20.min(len / 3);
    for p in 2..=max_period {
        let pattern = &bytes[..p];
        let mut reps = 1;
        let mut pos = p;
        while pos + p <= len && &bytes[pos..pos + p] == pattern {
            reps += 1;
            pos += p;
        }
        if reps >= 3 && (reps * p) * 100 >= len * 70 {
            return true;
        }
    }
    false
}

/// Check if output has character-level n-gram repetition
///
/// Checks the full output string and each individual word (for words
/// with length >= 6) to catch patterns like "foo VILLEVILLEVILLE bar".
fn has_char_ngram_repetition(output: &str) -> bool {
    if check_substring_repetition(output) {
        return true;
    }
    output
        .split_whitespace()
        .any(|word| word.len() >= 6 && check_substring_repetition(word))
}

/// Check if words contain a 2-word repeating pattern
///
/// Returns true if a 2-word bigram repeats for at least half the chunks.
fn has_two_word_repetition(words: &[&str]) -> bool {
    if words.len() < 6 {
        return false;
    }
    let pattern: Vec<_> = words.iter().take(2).collect();
    let matches = words
        .chunks(2)
        .filter(|chunk| chunk.len() == 2 && chunk[0] == *pattern[0] && chunk[1] == *pattern[1])
        .count();
    matches >= words.len() / 2 / 2
}

/// Check if all words in a slice are identical
fn all_words_identical(words: &[&str]) -> bool {
    let first = words.first();
    first.is_some() && words.iter().all(|w| Some(w) == first)
}

/// Check if output is highly repetitive
fn is_repetitive(output: &str) -> bool {
    // Character-level n-gram check catches patterns like "VILLEVILLEVILLE"
    // that word-level checks miss (single continuous token, no whitespace).
    if has_char_ngram_repetition(output) {
        return true;
    }

    let words: Vec<&str> = output.split_whitespace().collect();
    if words.len() < 5 {
        return false;
    }

    all_words_identical(&words) || has_two_word_repetition(&words)
}

/// Truncate string for display (UTF-8 safe)
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find last valid char boundary at or before max_len
        let end = s[..=max_len.min(s.len())]
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= max_len)
            .last()
            .unwrap_or(0);
        format!("{}...", &s[..end])
    }
}

#[cfg(test)]
#[path = "oracle_tests.rs"]
mod oracle_tests;
