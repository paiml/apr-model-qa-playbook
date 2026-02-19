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

// --- Additional mutation-killing tests ---

// Mutation: right != 0 → false (division by zero guard)
#[test]
fn test_arithmetic_division_nonzero_denominator() {
    let oracle = ArithmeticOracle::new();
    // 20/4 should correctly evaluate to 5
    let result = oracle.evaluate("20/4=", "5");
    assert!(result.is_corroborated());
    // 20/4 should NOT match 6 (wrong answer)
    let wrong = oracle.evaluate("20/4=", "6");
    assert!(wrong.is_falsified());
}

// Mutation: || → && in CodeSyntaxOracle (has_code_pattern || output.len() < 20)
#[test]
fn test_code_syntax_short_output_without_patterns() {
    let oracle = CodeSyntaxOracle::new();
    // Short output (< 20 chars) without code patterns should still corroborate
    let result = oracle.evaluate("test", "short text here");
    assert!(result.is_corroborated());
}

#[test]
fn test_code_syntax_long_output_with_patterns() {
    let oracle = CodeSyntaxOracle::new();
    // Long output (>= 20 chars) with code patterns should corroborate
    let result = oracle.evaluate("test", "def foo(): return 42 end");
    assert!(result.is_corroborated());
}

// Mutation: == → != in is_repetitive (words.iter().all(|w| Some(w) == first))
#[test]
fn test_is_repetitive_same_word_repeated() {
    // All same words should be flagged as repetitive
    assert!(is_repetitive("word word word word word word"));
    // Truly different words should NOT be flagged (12 unique words to avoid 2-word pattern)
    assert!(!is_repetitive(
        "one two three four five six seven eight nine ten eleven twelve"
    ));
}

#[test]
fn test_is_repetitive_mixed_words() {
    // If first word matches all others, it's repetitive
    assert!(is_repetitive("x x x x x"));
    // Completely different words should not trigger any pattern
    assert!(!is_repetitive("a b c d e f g h i j k l"));
}
