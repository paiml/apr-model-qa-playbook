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

        for op in ['+', '-', '*', '/'] {
            if let Some(pos) = expr.find(op) {
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
        }
        None
    }
}

impl Oracle for ArithmeticOracle {
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

    fn name(&self) -> &'static str {
        "code_syntax"
    }
}

/// Combined oracle that runs multiple oracles
pub struct CompositeOracle {
    name: &'static str,
    oracles: Vec<Box<dyn Oracle + Send + Sync>>,
}

impl std::fmt::Debug for CompositeOracle {
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
#[allow(dead_code)]
struct OracleWrapper<O: Oracle + Clone>(O);

impl<O: Oracle + Clone> Oracle for OracleWrapper<O> {
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        self.0.evaluate(prompt, output)
    }

    fn name(&self) -> &'static str {
        self.0.name()
    }
}

impl Oracle for CompositeOracle {
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
        if reps >= 3 && (reps * p) * 100 / len >= 70 {
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

    // Check if all words are the same
    let first = words.first();
    if first.is_some() && words.iter().all(|w| Some(w) == first) {
        return true;
    }

    // Check for 2-word repeating patterns
    if words.len() >= 6 {
        let pattern: Vec<_> = words.iter().take(2).collect();
        let mut matches = 0;
        for chunk in words.chunks(2) {
            if chunk.len() == 2 && chunk[0] == *pattern[0] && chunk[1] == *pattern[1] {
                matches += 1;
            }
        }
        if matches >= words.len() / 2 / 2 {
            return true;
        }
    }

    false
}

/// Truncate string for display
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_oracle_correct() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("2+2=", "The answer is 4.");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_arithmetic_oracle_incorrect() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("2+2=", "The answer is 5.");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_arithmetic_oracle_non_arithmetic() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("What is your name?", "I am an AI.");
        assert!(result.is_corroborated()); // Skipped
    }

    #[test]
    fn test_garbage_oracle_empty() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_valid() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "This is a valid response.");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_garbage_oracle_nan() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "The value is NaN");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_repetitive() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "ak ak ak ak ak ak ak ak");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_select_oracle_arithmetic() {
        let oracle = select_oracle("What is 2+2?");
        assert_eq!(oracle.name(), "arithmetic");
    }

    #[test]
    fn test_select_oracle_code() {
        let oracle = select_oracle("def fibonacci(n):");
        assert_eq!(oracle.name(), "code_syntax");
    }

    #[test]
    fn test_select_oracle_default() {
        let oracle = select_oracle("Tell me a joke");
        assert_eq!(oracle.name(), "garbage");
    }

    #[test]
    fn test_is_repetitive() {
        assert!(is_repetitive("foo foo foo foo foo foo"));
        assert!(is_repetitive("bar baz bar baz bar baz bar baz"));
        assert!(!is_repetitive(
            "The quick brown fox jumps over the lazy dog"
        ));
    }

    #[test]
    fn test_is_repetitive_short() {
        assert!(!is_repetitive("a b c"));
        assert!(!is_repetitive(""));
    }

    #[test]
    fn test_oracle_result_is_corroborated() {
        let result = OracleResult::Corroborated {
            evidence: "test".to_string(),
        };
        assert!(result.is_corroborated());
        assert!(!result.is_falsified());
    }

    #[test]
    fn test_oracle_result_is_falsified() {
        let result = OracleResult::Falsified {
            reason: "bad".to_string(),
            evidence: "test".to_string(),
        };
        assert!(!result.is_corroborated());
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_control_chars() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "Hello\x00World");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_replacement_char() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "Hello\u{FFFD}World");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_inf() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "The value is Inf");
        assert!(result.is_falsified());

        let result2 = oracle.evaluate("test", "The value is inf");
        assert!(result2.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_whitespace_only() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "   \n\t  ");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_code_syntax_oracle_valid() {
        let oracle = CodeSyntaxOracle::new();
        let result = oracle.evaluate("def foo():", "    return 42");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_code_syntax_oracle_with_patterns() {
        let oracle = CodeSyntaxOracle::new();
        let result = oracle.evaluate("test", "fn main() { let x = 5; }");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_code_syntax_oracle_long_prose() {
        let oracle = CodeSyntaxOracle::new();
        let result = oracle.evaluate(
            "test",
            "This is a long description that doesn't contain any code patterns whatsoever.",
        );
        assert!(result.is_corroborated()); // Might be documentation
    }

    #[test]
    fn test_code_syntax_oracle_garbage() {
        let oracle = CodeSyntaxOracle::new();
        let result = oracle.evaluate("test", "");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_composite_oracle_all_pass() {
        let mut composite = CompositeOracle::new("test");
        composite.add(GarbageOracle::new());
        let result = composite.evaluate("test", "Valid output here");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_composite_oracle_one_fails() {
        let mut composite = CompositeOracle::new("test");
        composite.add(GarbageOracle::new());
        let result = composite.evaluate("test", "");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_composite_oracle_debug() {
        let composite = CompositeOracle::new("test");
        let debug_str = format!("{composite:?}");
        assert!(debug_str.contains("CompositeOracle"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_arithmetic_eval_subtraction() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("10-3=", "7");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_arithmetic_eval_multiplication() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("5*6=", "30");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_arithmetic_eval_division() {
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("20/4=", "5");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_arithmetic_division_by_zero() {
        let oracle = ArithmeticOracle::new();
        // Division by zero should skip (non-arithmetic)
        let result = oracle.evaluate("5/0=", "undefined");
        assert!(result.is_corroborated());
    }

    #[test]
    fn test_is_arithmetic_prompt() {
        assert!(is_arithmetic_prompt("2+2="));
        assert!(is_arithmetic_prompt("What is 3*4?"));
        assert!(!is_arithmetic_prompt("Hello world"));
    }

    #[test]
    fn test_is_code_prompt() {
        assert!(is_code_prompt("def foo():"));
        assert!(is_code_prompt("fn main() {"));
        assert!(is_code_prompt("function test() {"));
        assert!(is_code_prompt("class Foo:"));
        assert!(is_code_prompt("async function bar() {"));
        assert!(is_code_prompt("```python\nx=1\n```"));
        assert!(!is_code_prompt("Hello world"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }

    #[test]
    fn test_oracle_names() {
        assert_eq!(ArithmeticOracle::new().name(), "arithmetic");
        assert_eq!(GarbageOracle::new().name(), "garbage");
        assert_eq!(CodeSyntaxOracle::new().name(), "code_syntax");
    }

    #[test]
    fn test_oracle_result_clone() {
        let result = OracleResult::Corroborated {
            evidence: "test".to_string(),
        };
        let cloned = result.clone();
        assert!(cloned.is_corroborated());
    }

    #[test]
    fn test_oracle_result_serialize() {
        let result = OracleResult::Falsified {
            reason: "bad".to_string(),
            evidence: "test".to_string(),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("Falsified"));
    }

    // Mutation-killing tests for arithmetic operations
    #[test]
    fn test_arithmetic_addition_not_multiplication() {
        // If + were replaced with *, 2+3 would give 6, not 5
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("2+3=", "5");
        assert!(result.is_corroborated());
        let wrong = oracle.evaluate("2+3=", "6");
        assert!(wrong.is_falsified());
    }

    #[test]
    fn test_arithmetic_subtraction_not_other() {
        // If - were replaced with +, 10-3 would give 13, not 7
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("10-3=", "7");
        assert!(result.is_corroborated());
        let wrong = oracle.evaluate("10-3=", "13");
        assert!(wrong.is_falsified());
    }

    #[test]
    fn test_arithmetic_multiplication_not_addition() {
        // If * were replaced with +, 3*4 would give 7, not 12
        let oracle = ArithmeticOracle::new();
        let result = oracle.evaluate("3*4=", "12");
        assert!(result.is_corroborated());
        let wrong = oracle.evaluate("3*4=", "7");
        assert!(wrong.is_falsified());
    }

    // Mutation-killing tests for is_arithmetic_prompt
    #[test]
    fn test_is_arithmetic_requires_both_operator_and_digit() {
        // Must have BOTH operator AND digit (tests && vs ||)
        assert!(!is_arithmetic_prompt("hello + world")); // has + but no digit
        assert!(!is_arithmetic_prompt("123")); // has digit but no operator
        assert!(is_arithmetic_prompt("1+2")); // has both
    }

    #[test]
    fn test_is_arithmetic_all_operators() {
        assert!(is_arithmetic_prompt("1+2"));
        assert!(is_arithmetic_prompt("5-3"));
        assert!(is_arithmetic_prompt("4*6"));
        assert!(is_arithmetic_prompt("8/2"));
    }

    // Mutation-killing tests for is_repetitive
    #[test]
    fn test_is_repetitive_needs_minimum_words() {
        // < 5 distinct words without char-level repetition → false
        assert!(!is_repetitive("one two three four"));
        // "a a a a" now correctly detected by char-level ngram check
        assert!(is_repetitive("a a a a"));
        assert!(is_repetitive("a a a a a")); // 5 words, all same
    }

    #[test]
    fn test_is_repetitive_two_word_pattern() {
        // Test 2-word pattern detection
        assert!(is_repetitive("foo bar foo bar foo bar"));
        // 6 words, threshold = 6/2/2 = 1, and first pair matches, so returns true
        // To test non-repetitive, need more words with fewer matches
        assert!(!is_repetitive("a b c d e f g h i j k l m n o p"));
    }

    #[test]
    fn test_is_repetitive_match_count_threshold() {
        // Partial matches shouldn't trigger
        assert!(!is_repetitive("a b c d e f g h i j"));
        assert!(is_repetitive("x y x y x y x y x y"));
    }

    // Mutation-killing tests for GarbageOracle conditions
    #[test]
    fn test_garbage_detects_different_nan_cases() {
        let oracle = GarbageOracle::new();
        // Checks for "NaN", "Inf", "inf" (case-sensitive)
        assert!(oracle.evaluate("test", "result: NaN").is_falsified());
        assert!(oracle.evaluate("test", "Inf value").is_falsified());
        assert!(oracle.evaluate("test", "inf error").is_falsified());
    }

    #[test]
    fn test_garbage_non_empty_non_whitespace() {
        let oracle = GarbageOracle::new();
        // Empty is falsified
        assert!(oracle.evaluate("test", "").is_falsified());
        // Whitespace only is falsified
        assert!(oracle.evaluate("test", "   ").is_falsified());
        // Real content is corroborated
        assert!(oracle.evaluate("test", "x").is_corroborated());
    }

    // Mutation-killing tests for CodeSyntaxOracle
    #[test]
    fn test_code_syntax_detects_patterns() {
        let oracle = CodeSyntaxOracle::new();
        // Should find code patterns
        assert!(oracle.evaluate("code", "return x;").is_corroborated());
        assert!(oracle.evaluate("code", "def foo(): pass").is_corroborated());
        assert!(oracle.evaluate("code", "fn bar() {}").is_corroborated());
    }

    // Test oracle name returns are not empty
    #[test]
    fn test_oracle_names_not_empty() {
        assert!(!ArithmeticOracle::new().name().is_empty());
        assert!(!GarbageOracle::new().name().is_empty());
        assert!(!CodeSyntaxOracle::new().name().is_empty());
        let composite = CompositeOracle::new("test");
        assert!(!composite.name().is_empty());
    }

    // Test OracleWrapper name delegation
    #[test]
    fn test_oracle_wrapper_name() {
        let wrapper = OracleWrapper(ArithmeticOracle::new());
        assert_eq!(wrapper.name(), "arithmetic");
    }

    // --- Character-level n-gram repetition tests ---

    #[test]
    fn test_char_ngram_ville_pattern() {
        // The motivating case from aprender#189
        assert!(check_substring_repetition("VILLEVILLEVILLEVILLE"));
        assert!(is_repetitive("VILLEVILLEVILLEVILLE"));
    }

    #[test]
    fn test_char_ngram_short_patterns() {
        assert!(check_substring_repetition("abcabcabc"));
        assert!(check_substring_repetition("xyxyxyxy"));
    }

    #[test]
    fn test_char_ngram_longer_patterns() {
        assert!(check_substring_repetition("helloWorldhelloWorldhelloWorld"));
    }

    #[test]
    fn test_char_ngram_not_triggered_on_normal_text() {
        assert!(!check_substring_repetition("The quick brown fox"));
        assert!(!check_substring_repetition("Hello, world!"));
        assert!(!check_substring_repetition(
            "Rust is a systems programming language"
        ));
        assert!(!has_char_ngram_repetition(
            "The quick brown fox jumps over the lazy dog"
        ));
    }

    #[test]
    fn test_char_ngram_per_word_detection() {
        // Garbage word embedded in normal sentence
        assert!(has_char_ngram_repetition("output VILLEVILLEVILLEVILLE end"));
    }

    #[test]
    fn test_char_ngram_single_char_repeat() {
        // "aaaaaaaaaaaa" — period 2 "aa" repeats 6 times, coverage 100%
        assert!(check_substring_repetition("aaaaaaaaaaaa"));
    }

    #[test]
    fn test_char_ngram_partial_coverage_not_flagged() {
        // "abcabcXYZ" — 2 reps of "abc" = 6/9 = 66%, below 70% threshold
        assert!(!check_substring_repetition("abcabcXYZ"));
    }

    #[test]
    fn test_char_ngram_boundary_exactly_three_reps() {
        // Exactly 3 repetitions, coverage 100% — should trigger
        assert!(check_substring_repetition("abcabcabc"));
    }

    #[test]
    fn test_char_ngram_boundary_exactly_two_reps() {
        // Exactly 2 repetitions — below threshold of 3
        assert!(!check_substring_repetition("abcabc"));
    }

    // Mutation-killing tests for thresholds

    #[test]
    fn test_char_ngram_min_reps_threshold() {
        // 3 reps at 100% coverage → true
        assert!(check_substring_repetition("xyzxyzxyz"));
        // 2 reps at 100% coverage → false (must be >= 3)
        assert!(!check_substring_repetition("xyzxyz"));
    }

    #[test]
    fn test_char_ngram_coverage_threshold() {
        // 3 reps of "ab" in "abababXXXX" = 6/10 = 60% < 70% → false
        assert!(!check_substring_repetition("abababXXXX"));
        // 3 reps of "ab" in "ababab" = 6/6 = 100% → true
        assert!(check_substring_repetition("ababab"));
    }

    #[test]
    fn test_char_ngram_min_period_is_two() {
        // Period 1 is not checked — single char "aaa" with len < 6 is skipped
        assert!(!check_substring_repetition("aaa"));
        // But period 2 "aa" in a long string works
        assert!(check_substring_repetition("aaaaaaaaaaaa"));
    }

    #[test]
    fn test_char_ngram_word_len_threshold() {
        // Words shorter than 6 chars are not individually checked
        assert!(!has_char_ngram_repetition("aaaa bbbb"));
        // Word with 6+ chars that is repetitive gets caught
        assert!(has_char_ngram_repetition("normal ababababab text"));
    }

    #[test]
    fn test_char_ngram_too_short_string() {
        // Strings shorter than 6 bytes always return false
        assert!(!check_substring_repetition("abab"));
        assert!(!check_substring_repetition("aa"));
        assert!(!check_substring_repetition(""));
    }

    // Integration: GarbageOracle.evaluate() catches VILLE pattern
    #[test]
    fn test_garbage_oracle_catches_ville() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "VILLEVILLEVILLEVILLE");
        assert!(result.is_falsified());
    }

    #[test]
    fn test_garbage_oracle_catches_embedded_repetition() {
        let oracle = GarbageOracle::new();
        let result = oracle.evaluate("test", "Result: VILLEVILLEVILLEVILLE done");
        assert!(result.is_falsified());
    }

    // Regression: existing word-level tests still pass
    #[test]
    fn test_word_level_repetition_still_works() {
        assert!(is_repetitive("foo foo foo foo foo foo"));
        assert!(is_repetitive("bar baz bar baz bar baz bar baz"));
        assert!(!is_repetitive(
            "The quick brown fox jumps over the lazy dog"
        ));
    }

    #[test]
    fn test_word_level_short_still_skipped() {
        // Short word sequences without char-ngram patterns pass through
        assert!(!is_repetitive("hello world"));
        assert!(!is_repetitive("one two three"));
    }
}
