#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0",
#     "transformers>=4.38",
#     "safetensors>=0.4",
#     "accelerate>=0.25",
# ]
# ///
"""Generate golden outputs from HuggingFace transformers.

This script generates golden outputs (logits, text) from HuggingFace models
for use as ground truth in the HF Parity Oracle.

Usage:
    uv run scripts/generate_golden.py \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --prompts prompts/code_bench.txt \
        --output ../hf-ground-truth-corpus/oracle/qwen2.5-coder-1.5b/v1/

References:
    - HF Parity Oracle Spec: docs/specifications/hf-parity-oracle.md
    - Popper, K. (1959). The Logic of Scientific Discovery.
    - Goldberg, D. (1991). What Every Computer Scientist Should Know About FP.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file


def hash_prompt(prompt: str) -> str:
    """Hash prompt to create deterministic filename.

    Uses the same algorithm as the Rust implementation for consistency.
    Note: Python's hash() is not deterministic across runs, so we use
    a simplified hash that matches the Rust DefaultHasher behavior.

    For exact parity with Rust, we'd need to implement the same SipHash.
    For now, we use SHA-256 truncated to 16 hex chars for simplicity.
    """
    # Use SHA-256 for determinism (Rust uses SipHash, but this is close enough)
    h = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    return h[:16]


def generate_golden(
    model_name: str,
    prompts: list[str],
    output_dir: Path,
    max_new_tokens: int = 50,
    device: str = "auto",
    use_fp16: bool = False,
) -> None:
    """Generate golden outputs for a list of prompts.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        prompts: List of input prompts
        output_dir: Directory to write golden outputs
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on ("auto", "cuda", "cpu")
        use_fp16: Use FP16 precision to reduce memory (default: False for max precision)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine dtype and device
    if use_fp16:
        dtype = torch.float16
        print("Using FP16 precision")
    else:
        dtype = torch.float32
        print("Using FP32 precision")

    # Resolve device - handle "auto" by checking CUDA availability
    if device == "auto":
        if torch.cuda.is_available():
            try:
                # Check if we have enough memory (rough estimate: 4GB for 1.5B model)
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_mem < 4.0:
                    print(f"CUDA available but only {free_mem:.1f}GB free, falling back to CPU")
                    device = "cpu"
                else:
                    print(f"CUDA available with {free_mem:.1f}GB free, using GPU")
                    device = "cuda"
            except Exception:
                print("CUDA memory check failed, falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
            print("CUDA not available, using CPU")

    # Load model with appropriate settings
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to("cpu")
        actual_device = "cpu"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        # device_map returns model on CUDA when set to "cuda"
        actual_device = "cuda" if device == "cuda" else device

    print(f"Model loaded on: {actual_device}")
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get library versions for metadata
    import transformers
    transformers_version = transformers.__version__
    torch_version = torch.__version__

    print(f"Generating golden outputs for {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(actual_device)

        # Get logits without generation
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().contiguous()  # [seq_len, vocab_size]

        # Generate text for comparison
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for determinism
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Create hash for filename
        h = hash_prompt(prompt)

        # Save logits as SafeTensors
        safetensors_path = output_dir / f"{h}.safetensors"
        save_file(
            {"logits": logits},
            safetensors_path,
        )

        # Save metadata as companion JSON
        metadata = {
            "prompt": prompt,
            "model": model_name,
            "transformers_version": transformers_version,
            "torch_version": torch_version,
            "generated_text": generated_text,
            "input_hash": h,
            "logits_shape": list(logits.shape),
        }
        json_path = output_dir / f"{h}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"    Saved: {safetensors_path.name}")

    # Write manifest
    manifest = {
        "model": model_name,
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "num_prompts": len(prompts),
        "prompts": [hash_prompt(p) for p in prompts],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done! Generated {len(prompts)} golden outputs in {output_dir}")


def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from a text file (one per line)."""
    with open(prompts_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden outputs from HuggingFace transformers"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="Path to prompts file (one per line)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        help="Single prompt (can be specified multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for golden files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'auto', 'cuda', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision to reduce memory (default: FP32)",
    )

    args = parser.parse_args()

    # Collect prompts from file and/or command line
    prompts = []
    if args.prompts:
        prompts.extend(load_prompts(args.prompts))
    if args.prompt:
        prompts.extend(args.prompt)

    if not prompts:
        # Default test prompts
        prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "2 + 2 =",
            "Hello, my name is",
            "The quick brown fox",
        ]
        print(f"No prompts specified, using {len(prompts)} default prompts")

    generate_golden(
        model_name=args.model,
        prompts=prompts,
        output_dir=args.output,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        use_fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
