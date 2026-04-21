"""
Project-level pytest configuration.

CLI options for real-world tests:

  --base-model-id   HuggingFace repo ID or local path for the base model.
                    e.g.  --base-model-id Qwen/Qwen3-1.7B
                    e.g.  --base-model-id /data/models/Qwen3-1.7B

  --adapter         NAME:PATH pair (repeatable).
                    e.g.  --adapter diagnosis:./adapters/implementation
                          --adapter implementation:./adapters/implementation

  --classifier      Path to the trained MLP classifier checkpoint (optional).
                    When omitted, a keyword-based stub router is used instead.

  --max-tokens      Maximum new tokens per generation (default 128).

Usage example:
  pytest test/test_real_world.py \\
    --base-model-id Qwen/Qwen3-1.7B \\
    --adapter diagnosis:./adapters/diagnosis \\
    --adapter implementation:./adapters/implementation \\
    -v -s
"""


def pytest_addoption(parser):
    parser.addoption(
        "--base-model-id",
        default=None,
        help="HuggingFace repo ID or local path for the base Qwen3 model.",
    )
    parser.addoption(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="LoRA adapter as NAME:PATH (repeatable).",
    )
    parser.addoption(
        "--classifier",
        default=None,
        metavar="PATH",
        help="Path to trained MLP classifier checkpoint (optional).",
    )
    parser.addoption(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens per generation request (default 128).",
    )
    parser.addoption(
        "--batch-size",
        type=int,
        default=4,
        help="Number of requests per Prefill batch in benchmark tests (default 4).",
    )
    parser.addoption(
        "--num-runs",
        type=int,
        default=3,
        help="Number of benchmark repetitions to average over (default 3).",
    )
    parser.addoption(
        "--backend",
        default="ours",
        choices=["ours", "peft", "vllm"],
        help="Which serving backend to test: ours | peft | vllm (default: ours).",
    )
    parser.addoption(
        "--num-adapters",
        type=int,
        default=None,
        help="Max number of adapters to load (default: all provided via --adapter).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_world: marks tests that require a real GPU and model weights",
    )
