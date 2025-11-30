# Fast-dLLM-Trida

Fast inference and evaluation framework for **Trida-7B**, a diffusion-based language model that uses masked decoding for accelerated text generation.

Built upon [NVIDIA's Fast-dLLM](https://github.com/NVlabs/Fast-dLLM).

## Overview

Trida is a 7-billion parameter diffusion language model that generates text through iterative refinement of masked tokens, rather than traditional autoregressive (left-to-right) generation. This approach enables parallel decoding within blocks for faster inference.

### Key Features

- **Masked Decoding Model (MDM)**: Iteratively refines masked tokens using confidence-based unmasking
- **Block-wise Parallelism**: Processes 32-token blocks in parallel for accelerated generation
- **Flexible Evaluation**: Integrated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for benchmarking
- **Interactive Interfaces**: Gradio web UI with real-time denoising visualization and CLI chatbot

## Installation

```bash
cd trida
pip install -e .
```

### Optional Dependencies

```bash
# For Gradio web interface
pip install -e ".[gradio]"

# For vLLM integration
pip install -e ".[vllm]"

# For Flash Attention
pip install -e ".[flash_attn]"

# For multimodal support
pip install -e ".[multimodal]"
```

## Quick Start

### Interactive Chat (CLI)

```bash
cd trida
python run_chatbot.py
```

### Interactive Chat (Web UI)

```bash
cd trida
python app.py
# Opens Gradio interface at http://localhost:10086
```

The web UI features real-time visualization of the denoising process, showing how masked tokens are progressively refined into generated text.

### Benchmark Evaluation

```bash
cd trida

# Single GPU
python eval.py \
    --model fast_dllm_trida \
    --tasks gsm8k \
    --batch_size 4 \
    --model_args model_path=trillionlabs/Trida-7B,threshold=0.9

# Multi-GPU with accelerate
accelerate launch eval.py \
    --model fast_dllm_trida \
    --tasks gsm8k \
    --batch_size 4
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 32 | Number of tokens generated per block |
| `small_block_size` | 8 | Refinement block size for iterative unmasking |
| `threshold` | 0.9 | Confidence threshold for unmasking tokens (0.5-1.0) |
| `max_new_tokens` | 2048 | Maximum number of tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling parameter |
| `use_block_cache` | False | Enable KV cache optimization |

## Supported Benchmarks

The evaluation harness supports various benchmarks including:

- **Math Reasoning**: GSM8K, MATH, HRM8K
- **Instruction Following**: IFEval
- **Language Understanding**: KoBEST (Korean), and other lm-eval tasks

Example evaluation commands are provided in `trida/eval_script.sh`.

## Project Structure

```
Fast-dLLM-Trida/
├── README.md
├── LICENSE                      # Apache 2.0
└── trida/
    ├── eval.py                  # Evaluation harness for lm-eval
    ├── app.py                   # Gradio web interface
    ├── run_chatbot.py           # CLI chatbot
    ├── generation_functions.py  # Core MDM sampling algorithms
    ├── eval_script.sh           # Example evaluation commands
    ├── requirements.txt
    ├── setup.py
    └── src/
        └── lmflow/              # LMFlow framework
            ├── models/          # Model wrappers
            ├── pipeline/        # Training/inference pipelines
            ├── optim/           # Optimizers (Adam, Sophia, etc.)
            └── utils/           # Utilities and conversation templates
```

## How Trida Works

1. **Input Processing**: The prompt is tokenized and prepared with special tokens
2. **Block Generation**: Text is generated in blocks of `block_size` tokens
3. **Masked Initialization**: Each block starts with all positions masked (`<|mask|>` token)
4. **Iterative Refinement**: Within each block:
   - Model predicts tokens for all masked positions
   - Tokens with confidence above `threshold` are unmasked
   - Process repeats until all tokens are unmasked or max iterations reached
5. **Completion**: Generation stops when `<|im_end|>` token is produced

## Model

The default model is [trillionlabs/Trida-7B-Preview](https://huggingface.co/trillionlabs/Trida-7B-Preview) on Hugging Face.

## Acknowledgments

- [NVIDIA Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) - Base framework
- [LMFlow](https://github.com/OptimalScale/LMFlow) - Training and inference utilities
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Evaluation framework

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
