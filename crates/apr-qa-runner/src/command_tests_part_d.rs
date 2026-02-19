#[test]
fn test_mock_runner_inference_tps_in_output() {
    let runner = MockCommandRunner::new().with_tps(55.3);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "Hello", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("55.3"));
}
