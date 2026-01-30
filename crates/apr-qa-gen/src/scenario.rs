//! QA Scenario generation
//!
//! Defines test scenarios for model qualification using property-based testing.

use crate::models::ModelId;
use crate::oracle::{OracleResult, select_oracle};
use serde::{Deserialize, Serialize};

/// Inference modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Direct inference via `apr run`
    Run,
    /// Interactive chat via `apr chat`
    Chat,
    /// HTTP server via `apr serve`
    Serve,
}

impl Modality {
    /// Get all modalities
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Run, Self::Chat, Self::Serve]
    }

    /// Get the apr command for this modality
    #[must_use]
    pub const fn command(&self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Chat => "chat",
            Self::Serve => "serve",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Run => write!(f, "run"),
            Self::Chat => write!(f, "chat"),
            Self::Serve => write!(f, "serve"),
        }
    }
}

/// Compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// CPU with SIMD acceleration
    Cpu,
    /// GPU with CUDA acceleration
    Gpu,
}

impl Backend {
    /// Get all backends
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Cpu, Self::Gpu]
    }

    /// Get the CLI flag for this backend
    #[must_use]
    pub const fn flag(&self) -> &'static str {
        match self {
            Self::Cpu => "",
            Self::Gpu => "--gpu",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Model format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// GGUF quantized format
    Gguf,
    /// `HuggingFace` `SafeTensors` format
    SafeTensors,
    /// Native APR format
    Apr,
}

impl Format {
    /// Get all formats
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Gguf, Self::SafeTensors, Self::Apr]
    }

    /// Get the file extension for this format
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Gguf => ".gguf",
            Self::SafeTensors => ".safetensors",
            Self::Apr => ".apr",
        }
    }

    /// Get the class (A=quantized, B=full precision)
    #[must_use]
    pub const fn class(&self) -> char {
        match self {
            Self::Gguf | Self::Apr => 'A', // Quantized
            Self::SafeTensors => 'B',      // Full precision
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "gguf"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::Apr => write!(f, "apr"),
        }
    }
}

/// Trace level for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceLevel {
    /// No tracing
    None,
    /// Basic timing and token counts
    Basic,
    /// Per-layer statistics
    Layer,
    /// Full tensor values
    Payload,
}

impl TraceLevel {
    /// Get all trace levels
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::None, Self::Basic, Self::Layer, Self::Payload]
    }

    /// Get the CLI value for this trace level
    #[must_use]
    pub const fn value(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Basic => "basic",
            Self::Layer => "layer",
            Self::Payload => "payload",
        }
    }
}

/// A single QA test scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaScenario {
    /// Unique scenario ID
    pub id: String,
    /// Model to test
    pub model: ModelId,
    /// Inference modality
    pub modality: Modality,
    /// Compute backend
    pub backend: Backend,
    /// Model format
    pub format: Format,
    /// Test prompt
    pub prompt: String,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Trace level
    pub trace_level: TraceLevel,
    /// Expected oracle type
    pub oracle_type: String,
}

impl QaScenario {
    /// Create a new scenario
    #[must_use]
    pub fn new(
        model: ModelId,
        modality: Modality,
        backend: Backend,
        format: Format,
        prompt: String,
        seed: u64,
    ) -> Self {
        let oracle = select_oracle(&prompt);
        Self {
            id: format!(
                "{}_{}_{}_{}_{:016x}",
                model.name, modality, backend, format, seed
            ),
            model,
            modality,
            backend,
            format,
            prompt,
            temperature: 0.0, // Deterministic by default
            max_tokens: 32,
            seed,
            trace_level: TraceLevel::None,
            oracle_type: oracle.name().to_string(),
        }
    }

    /// Set temperature
    #[must_use]
    pub const fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set trace level
    #[must_use]
    pub const fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    /// Generate the apr CLI command for this scenario
    #[must_use]
    pub fn to_command(&self, model_path: &str) -> String {
        let backend_flag = self.backend.flag();
        let trace_flag = if self.trace_level == TraceLevel::None {
            String::new()
        } else {
            format!("--trace --trace-level {}", self.trace_level.value())
        };

        match self.modality {
            Modality::Run => {
                format!(
                    "apr run {model_path} '{}' -n {} --seed {} --temperature {} {backend_flag} {trace_flag}",
                    escape_prompt(&self.prompt),
                    self.max_tokens,
                    self.seed,
                    self.temperature
                )
                .trim()
                .to_string()
            }
            Modality::Chat => {
                format!(
                    "echo '{}' | apr chat {model_path} --temperature {} {backend_flag} {trace_flag}",
                    escape_prompt(&self.prompt),
                    self.temperature
                )
                .trim()
                .to_string()
            }
            Modality::Serve => {
                format!(
                    r#"apr serve {model_path} --port ${{PORT}} {backend_flag} &
sleep 2
curl -s http://localhost:${{PORT}}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{{"prompt": "{}", "max_tokens": {}, "temperature": {}}}'
kill %1"#,
                    escape_json(&self.prompt),
                    self.max_tokens,
                    self.temperature
                )
            }
        }
    }

    /// Evaluate the output using the appropriate oracle
    #[must_use]
    pub fn evaluate(&self, output: &str) -> OracleResult {
        let oracle = select_oracle(&self.prompt);
        oracle.evaluate(&self.prompt, output)
    }

    /// Get the MQS category this scenario contributes to
    #[must_use]
    pub const fn mqs_category(&self) -> &'static str {
        match self.modality {
            Modality::Run => match self.backend {
                Backend::Cpu => "A1",
                Backend::Gpu => "A2",
            },
            Modality::Chat => match self.backend {
                Backend::Cpu => "A3",
                Backend::Gpu => "A4",
            },
            Modality::Serve => match self.backend {
                Backend::Cpu => "A5",
                Backend::Gpu => "A6",
            },
        }
    }
}

/// Escape a prompt for shell usage
fn escape_prompt(prompt: &str) -> String {
    prompt.replace('\'', "'\\''")
}

/// Escape a prompt for JSON usage
fn escape_json(prompt: &str) -> String {
    prompt
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Scenario generator for property-based testing
#[derive(Debug, Clone)]
pub struct ScenarioGenerator {
    /// Model to generate scenarios for
    pub model: ModelId,
    /// Number of scenarios per modality/backend/format combination
    pub scenarios_per_combination: usize,
    /// Prompts to use
    pub prompts: Vec<String>,
}

impl ScenarioGenerator {
    /// Create a new generator
    #[must_use]
    pub fn new(model: ModelId) -> Self {
        Self {
            model,
            scenarios_per_combination: 100,
            prompts: default_prompts(),
        }
    }

    /// Set the number of scenarios per combination
    #[must_use]
    pub const fn with_scenarios_per_combination(mut self, count: usize) -> Self {
        self.scenarios_per_combination = count;
        self
    }

    /// Set custom prompts
    #[must_use]
    pub fn with_prompts(mut self, prompts: Vec<String>) -> Self {
        self.prompts = prompts;
        self
    }

    /// Generate all scenarios for a model
    #[must_use]
    pub fn generate(&self) -> Vec<QaScenario> {
        let mut scenarios = Vec::new();
        let mut seed: u64 = 0;

        for modality in Modality::all() {
            for backend in Backend::all() {
                for format in Format::all() {
                    for i in 0..self.scenarios_per_combination {
                        let prompt_idx = i % self.prompts.len();
                        let prompt = &self.prompts[prompt_idx];

                        scenarios.push(QaScenario::new(
                            self.model.clone(),
                            modality,
                            backend,
                            format,
                            prompt.clone(),
                            seed,
                        ));

                        seed = seed.wrapping_add(1);
                    }
                }
            }
        }

        scenarios
    }

    /// Generate scenarios for a specific combination
    #[must_use]
    pub fn generate_for(
        &self,
        modality: Modality,
        backend: Backend,
        format: Format,
    ) -> Vec<QaScenario> {
        let mut scenarios = Vec::new();
        let base_seed: u64 = (modality as u64) << 32 | (backend as u64) << 16 | (format as u64);

        for (i, prompt) in self
            .prompts
            .iter()
            .enumerate()
            .take(self.scenarios_per_combination)
        {
            scenarios.push(QaScenario::new(
                self.model.clone(),
                modality,
                backend,
                format,
                prompt.clone(),
                base_seed.wrapping_add(i as u64),
            ));
        }

        scenarios
    }
}

/// Get default test prompts
fn default_prompts() -> Vec<String> {
    vec![
        // Arithmetic (verifiable)
        "What is 2+2?".to_string(),
        "Calculate 7*8".to_string(),
        "What is 15-7?".to_string(),
        "What is 100/4?".to_string(),
        "2+2=".to_string(),
        // Code completion
        "def fibonacci(n):".to_string(),
        "fn main() {".to_string(),
        "async function fetch() {".to_string(),
        "class Person:".to_string(),
        // Instruction following
        "Write a haiku about programming.".to_string(),
        "List three colors.".to_string(),
        "Explain what a variable is in one sentence.".to_string(),
        "Say hello in three languages.".to_string(),
        // Edge cases
        String::new(),            // Empty prompt
        " ".to_string(),          // Whitespace only
        "Hello!".to_string(),     // Simple greeting
        "你好".to_string(),       // Chinese
        "こんにちは".to_string(), // Japanese
    ]
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::redundant_closure_for_method_calls)]
mod tests {
    use super::*;
    use crate::models::ModelId;

    #[test]
    fn test_scenario_creation() {
        let model = ModelId::new("Qwen", "Qwen2.5-Coder-1.5B");
        let scenario = QaScenario::new(
            model.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            42,
        );

        assert_eq!(scenario.modality, Modality::Run);
        assert_eq!(scenario.backend, Backend::Cpu);
        assert_eq!(scenario.format, Format::Gguf);
        assert_eq!(scenario.oracle_type, "arithmetic");
    }

    #[test]
    fn test_scenario_to_command_run() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            0,
        );

        let cmd = scenario.to_command("model.gguf");
        assert!(cmd.contains("apr run"));
        assert!(cmd.contains("model.gguf"));
        assert!(cmd.contains("Hello"));
    }

    #[test]
    fn test_scenario_to_command_gpu() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Gpu,
            Format::Gguf,
            "Hello".to_string(),
            0,
        );

        let cmd = scenario.to_command("model.gguf");
        assert!(cmd.contains("--gpu"));
    }

    #[test]
    fn test_scenario_generator() {
        let model = ModelId::new("test", "model");
        let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(10);

        let scenarios = generator.generate();

        // 3 modalities × 2 backends × 3 formats × 10 = 180
        assert_eq!(scenarios.len(), 180);
    }

    #[test]
    fn test_scenario_mqs_category() {
        let model = ModelId::new("test", "model");

        let run_cpu = QaScenario::new(
            model.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(run_cpu.mqs_category(), "A1");

        let chat_gpu = QaScenario::new(
            model.clone(),
            Modality::Chat,
            Backend::Gpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(chat_gpu.mqs_category(), "A4");

        let serve_cpu = QaScenario::new(
            model,
            Modality::Serve,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(serve_cpu.mqs_category(), "A5");
    }

    #[test]
    fn test_format_class() {
        assert_eq!(Format::Gguf.class(), 'A');
        assert_eq!(Format::Apr.class(), 'A');
        assert_eq!(Format::SafeTensors.class(), 'B');
    }

    #[test]
    fn test_escape_prompt() {
        assert_eq!(escape_prompt("hello"), "hello");
        assert_eq!(escape_prompt("it's"), "it'\\''s");
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_json("say \"hi\""), "say \\\"hi\\\"");
    }

    #[test]
    fn test_escape_json_backslash() {
        assert_eq!(escape_json("path\\file"), "path\\\\file");
    }

    #[test]
    fn test_scenario_with_temperature() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        )
        .with_temperature(0.7);

        assert!((scenario.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scenario_with_max_tokens() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        )
        .with_max_tokens(256);

        assert_eq!(scenario.max_tokens, 256);
    }

    #[test]
    fn test_scenario_with_trace_level() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        )
        .with_trace_level(TraceLevel::Layer);

        assert_eq!(scenario.trace_level, TraceLevel::Layer);
    }

    #[test]
    fn test_scenario_to_command_chat() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            0,
        );

        let cmd = scenario.to_command("model.gguf");
        assert!(cmd.contains("apr chat"));
        assert!(cmd.contains("echo"));
    }

    #[test]
    fn test_scenario_to_command_serve() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Serve,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            0,
        );

        let cmd = scenario.to_command("model.gguf");
        assert!(cmd.contains("apr serve"));
        assert!(cmd.contains("curl"));
        assert!(cmd.contains("/v1/completions"));
    }

    #[test]
    fn test_scenario_to_command_with_trace() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            0,
        )
        .with_trace_level(TraceLevel::Payload);

        let cmd = scenario.to_command("model.gguf");
        assert!(cmd.contains("--trace"));
        assert!(cmd.contains("--trace-level payload"));
    }

    #[test]
    fn test_scenario_evaluate() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "What is 2+2?".to_string(),
            0,
        );

        let result = scenario.evaluate("The answer is 4");
        assert!(matches!(result, crate::OracleResult::Corroborated { .. }));
    }

    #[test]
    fn test_trace_level_value() {
        assert_eq!(TraceLevel::None.value(), "none");
        assert_eq!(TraceLevel::Basic.value(), "basic");
        assert_eq!(TraceLevel::Layer.value(), "layer");
        assert_eq!(TraceLevel::Payload.value(), "payload");
    }

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", Modality::Run), "run");
        assert_eq!(format!("{}", Modality::Chat), "chat");
        assert_eq!(format!("{}", Modality::Serve), "serve");
    }

    #[test]
    fn test_backend_flag() {
        assert_eq!(Backend::Cpu.flag(), "");
        assert_eq!(Backend::Gpu.flag(), "--gpu");
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Cpu), "cpu");
        assert_eq!(format!("{}", Backend::Gpu), "gpu");
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", Format::Gguf), "gguf");
        assert_eq!(format!("{}", Format::SafeTensors), "safetensors");
        assert_eq!(format!("{}", Format::Apr), "apr");
    }

    #[test]
    fn test_modality_all() {
        let all = Modality::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&Modality::Run));
        assert!(all.contains(&Modality::Chat));
        assert!(all.contains(&Modality::Serve));
    }

    #[test]
    fn test_backend_all() {
        let all = Backend::all();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&Backend::Cpu));
        assert!(all.contains(&Backend::Gpu));
    }

    #[test]
    fn test_format_all() {
        let all = Format::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&Format::Gguf));
        assert!(all.contains(&Format::SafeTensors));
        assert!(all.contains(&Format::Apr));
    }

    #[test]
    fn test_generator_with_prompts() {
        let model = ModelId::new("test", "model");
        let prompts = vec!["prompt1".to_string(), "prompt2".to_string()];
        let generator = ScenarioGenerator::new(model)
            .with_prompts(prompts.clone())
            .with_scenarios_per_combination(2);

        assert_eq!(generator.prompts, prompts);
    }

    #[test]
    fn test_generator_generate_for() {
        let model = ModelId::new("test", "model");
        let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(5);

        let scenarios = generator.generate_for(Modality::Run, Backend::Cpu, Format::Gguf);
        assert_eq!(scenarios.len(), 5);

        for s in &scenarios {
            assert_eq!(s.modality, Modality::Run);
            assert_eq!(s.backend, Backend::Cpu);
            assert_eq!(s.format, Format::Gguf);
        }
    }

    #[test]
    fn test_default_prompts_coverage() {
        let prompts = default_prompts();
        assert!(!prompts.is_empty());
        // Should have arithmetic prompts
        assert!(prompts.iter().any(|p| p.contains('+') || p.contains('*')));
        // Should have code prompts
        assert!(
            prompts
                .iter()
                .any(|p| p.starts_with("def ") || p.starts_with("fn "))
        );
        // Should have empty prompt for edge case
        assert!(prompts.iter().any(|p| p.is_empty()));
    }

    #[test]
    fn test_mqs_category_all_combinations() {
        let model = ModelId::new("test", "model");

        // Run GPU
        let run_gpu = QaScenario::new(
            model.clone(),
            Modality::Run,
            Backend::Gpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(run_gpu.mqs_category(), "A2");

        // Chat CPU
        let chat_cpu = QaScenario::new(
            model.clone(),
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(chat_cpu.mqs_category(), "A3");

        // Serve GPU
        let serve_gpu = QaScenario::new(
            model,
            Modality::Serve,
            Backend::Gpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );
        assert_eq!(serve_gpu.mqs_category(), "A6");
    }

    #[test]
    fn test_scenario_clone() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            42,
        );

        let cloned = scenario.clone();
        assert_eq!(cloned.id, scenario.id);
        assert_eq!(cloned.seed, scenario.seed);
    }

    #[test]
    fn test_scenario_serialize() {
        let model = ModelId::new("test", "model");
        let scenario = QaScenario::new(
            model,
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "test".to_string(),
            0,
        );

        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("\"modality\":\"run\""));
        assert!(json.contains("\"backend\":\"cpu\""));
    }
}
