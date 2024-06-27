# Socratic LLM

Using Large Language Models (LLMs) in education presents unique challenges. Typically, LLMs are designed to provide
direct answers to questions, which can hinder students' critical thinking and self-discovery skills. To address this, we
focus on fine-tuning LLMs to facilitate Socratic interactions. Instead of giving straightforward answers, these models
guide students to explore and find the answers themselves. We achieve this through Direct Preference Optimization (DPO).
We test our approach with diverse datasets, including various educational materials and Socratic dialogues. Using
advanced models like GPT-3.5 for evaluation, our results show that DPO successfully fine-tunes LLMs for Socratic
dialogue, enhancing their educational value.

This repository contains the source material for the paper "Fine Tuning a Large Language Model with DPO for
the Socratic method."

# Model inference

You can access the model directly from
HuggingFace [socratic-llm](https://huggingface.co/giovanni-gatti-pinheiro/socratic-llm).

```python
from peft import AutoPeftModelForCausalLM, PeftConfig
import torch
from transformers import AutoTokenizer

base_model = PeftConfig.from_pretrained("giovanni-gatti-pinheiro/socratic-llm").base_model_name_or_path
model = AutoPeftModelForCausalLM.from_pretrained(
    "giovanni-gatti-pinheiro/socratic-llm",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
```

Check HuggingFace's `transformers` library for more details.

# Scripts

We also make available evaluation scripts.

 - `eval_gpt_4o.py`: Perform evaluation of `GPT-4o` and prompt engineering (requires an OpenAI API key)
 - `eval_model.py`: Perform evaluation of the model with prompt engineering only or with the LoRA adapter

For each script, check `--help` for more details.

# Installing dependencies

You can run evaluation scripts either with a Docker image (see `Dockerfile`) or by installing the project dependencies 
with `requirements.txt`
