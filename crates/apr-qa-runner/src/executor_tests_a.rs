
use super::*;
use crate::command::MockCommandRunner;
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

fn test_playbook() -> Playbook {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

/// Create a temp file (file mode) for testing.
/// Returns (tempdir, file_path_string) - keep tempdir alive for test duration.
fn create_test_model_file(format: Format) -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().expect("failed to create temp directory for test model file");
    let filename = match format {
        Format::Gguf => "model.gguf",
        Format::Apr => "model.apr",
        Format::SafeTensors => "model.safetensors",
    };
    let file_path = tmp.path().join(filename);
    std::fs::write(&file_path, b"fake model data")
        .expect("failed to write fake model data to temp file");
    let path = file_path.to_string_lossy().to_string();
    (tmp, path)
}


include!("executor_tests_a_part_a.rs");

include!("executor_tests_a_part_b.rs");
