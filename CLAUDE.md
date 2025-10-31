# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The Language Model Evaluation Harness (lm-evaluation-harness) is a unified framework for testing generative language models on a large number of evaluation tasks. It's used by Hugging Face's Open LLM Leaderboard and is designed to be a standard, reproducible benchmark for LLM evaluation.

Version: 0.4.9.1

## Essential Commands

### Installation & Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies (for contributing)
pip install -e ".[dev]"

# Install with specific extras (e.g., vllm, api, wandb)
pip install -e ".[vllm,api,wandb]"
```

### Running Evaluations

```bash
# Basic evaluation (HuggingFace model)
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# List all available tasks
lm_eval --tasks list

# Evaluate multiple tasks
lm_eval --model hf \
    --model_args pretrained=model-name \
    --tasks lambada_openai,hellaswag,winogrande \
    --device cuda:0 \
    --batch_size auto

# Evaluate with logging
lm_eval --model hf \
    --model_args pretrained=model-name \
    --tasks hellaswag \
    --log_samples \
    --output_path results/
```

### Testing

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_evaluator.py

# Run with coverage
pytest --cov=lm_eval
```

### Development

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run linter (ruff)
ruff check lm_eval/

# Run formatter
ruff format lm_eval/
```

## Architecture

### Core Components

1. **`lm_eval/evaluator.py`**: Main evaluation orchestration logic
   - `simple_evaluate()`: Primary entry point for running evaluations
   - `evaluate()`: Lower-level evaluation function
   - Handles task execution, result aggregation, and caching

2. **`lm_eval/models/`**: Model adapters and backends
   - `huggingface.py`: HuggingFace transformers models
   - `vllm_causallms.py`: vLLM backend for fast inference
   - `sglang_causallms.py`: SGLang backend
   - `api_models.py`: Generic API-based models
   - `openai_completions.py`: OpenAI API models
   - `anthropic_llms.py`: Anthropic API models
   - Each model type inherits from `lm_eval.api.model.LM` base class

3. **`lm_eval/tasks/`**: Task definitions and configurations
   - Tasks are defined via YAML files
   - Each task folder contains: task YAML(s), optional `utils.py` for custom processing
   - Task registry automatically discovers tasks in this directory
   - Group configurations combine multiple tasks (e.g., MMLU with 57 subtasks)

4. **`lm_eval/api/`**: Core abstractions
   - `model.py`: Base `LM` class that all models inherit from
   - `task.py`: Task configuration and execution logic
   - `metrics.py`: Metric calculation functions
   - `registry.py`: Task and model registration system

5. **`lm_eval/filters/`**: Output processing filters
   - Post-processing of model outputs (e.g., extracting answers, cleaning text)

6. **`lm_eval/loggers/`**: Result tracking and logging
   - WandB integration
   - HuggingFace Hub logging
   - Evaluation tracking and result serialization

### Task System

Tasks are configured using YAML files with the following key components:

- **Dataset configuration**: `dataset_path`, `dataset_name`, `dataset_kwargs`
- **Prompt templates**: `doc_to_text`, `doc_to_target`, `doc_to_choice`
  - Supports Jinja2 templating
  - Can use Python functions via `!function` syntax
- **Metrics**: `metric_list` defines scoring metrics
- **Processing**: Optional `process_docs` function for dataset preprocessing
- **Few-shot**: `fewshot_split` or `fewshot_config` for few-shot examples

### Request Types

The evaluation harness supports three main request types:

1. **`loglikelihood`**: Compute log-likelihood of continuation given context (for multiple choice)
2. **`loglikelihood_rolling`**: Compute log-likelihood of entire sequence (for perplexity)
3. **`generate_until`**: Generate text until stopping criteria (for generative tasks)

## Creating New Tasks

### Basic Task Creation Workflow

1. Create task folder: `lm_eval/tasks/<task_name>/`
2. Create YAML config: `lm_eval/tasks/<task_name>/<task_name>.yaml`
3. (Optional) Create utils: `lm_eval/tasks/<task_name>/utils.py`
4. Test the task: `python -m scripts.write_out --tasks <task_name> --num_fewshot 5`
5. Add to `lm_eval/tasks/README.md`

### YAML Structure

```yaml
task: task_name
dataset_path: dataset_on_hf_hub
dataset_name: null  # or config name
test_split: test
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

### Using Custom Processing Functions

In `utils.py`:
```python
def process_docs(dataset):
    def _process_doc(doc):
        return {
            "query": doc["input"],
            "choices": doc["options"],
            "gold": int(doc["label"])
        }
    return dataset.map(_process_doc)
```

In YAML:
```yaml
process_docs: !function utils.process_docs
```

## Key Patterns & Conventions

### Model Arguments
Model-specific arguments are passed via `--model_args` as comma-separated key=value pairs:
```bash
--model_args pretrained=model-name,dtype=float16,device_map=auto,trust_remote_code=True
```

### Task Versioning
All tasks should include version metadata:
```yaml
metadata:
  version: 1.0
```
Increment version when making breaking changes to task configuration.

### Caching
- Use `--use_cache <path>` for caching model responses (avoids re-running expensive inference)
- Use `--cache_requests true` for caching dataset preprocessing
- Cache files stored in `lm_eval/cache/.cache` or custom path via `LM_HARNESS_CACHE_PATH`

### Batch Size
- Use `--batch_size auto` to automatically detect optimal batch size
- Use `--batch_size auto:N` to recompute batch size N times during evaluation (helpful when documents vary in length)

### Multi-GPU Evaluation
Three approaches supported:
1. Data parallelism: `accelerate launch -m lm_eval ...`
2. Model parallelism: `--model_args parallelize=True`
3. Combined: Both of the above together

## Important Files

- `lm_eval/__main__.py`: CLI entry point and argument parsing
- `lm_eval/evaluator.py`: Main evaluation logic
- `lm_eval/utils.py`: Shared utility functions
- `pyproject.toml`: Package configuration and dependencies
- `.pre-commit-config.yaml`: Pre-commit hooks (ruff, codespell, etc.)

## Testing Philosophy

- Tasks should be tested against reference implementations when available
- Use `--write_out` to inspect prompts and verify formatting
- Use `--limit 10` to test on small subset before full evaluation
- Set `LOGLEVEL=DEBUG` for verbose logging during development

## Common Pitfalls

1. **Prompt formatting**: Ensure `doc_to_text` and `doc_to_target` don't have unexpected trailing/leading whitespace
2. **Dataset splits**: Check that `test_split`, `validation_split` names match the actual dataset splits
3. **Metrics**: Verify metric names are valid (see `lm_eval/api/metrics.py` for supported metrics)
4. **Dependencies**: Task-specific dependencies go in `pyproject.toml` under `[project.optional-dependencies]`
5. **Version bumps**: Always increment task version when making changes that affect results
