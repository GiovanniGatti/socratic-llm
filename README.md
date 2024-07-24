<div align="center">
    <img alt="ollama" height="250px" src="./resources/banner.png">
</div>

# Socratic LLM

[![Static Badge](https://img.shields.io/badge/Model%20-%20%40%20HuggingFace%20-blue?style=flat&logo=huggingface&logoSize=20px&color=blue&link=https%3A%2F%2Fhuggingface.co%2Feurecom-ds%2FPhi-3-mini-4k-socratic)](https://huggingface.co/eurecom-ds/Phi-3-mini-4k-socratic)
[![Static Badge](https://img.shields.io/badge/Model%20-%20%40%20Ollama%20-blue?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAACgAAAA5CAYAAABEdGlTAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAACxMAAAsTAQCanBgAAAiASURBVGhD7ZhtaJ1nHcZPTtI2scli0tF0ypgyWFfndKUaV62CkymVyUTGnBsyvwymIpN9KEMpG4KK%2BkFlIMXVbvVDwW5MXeYKVYn7sGKb1s4NVjBa0yZrXprknDTN%2B5vX79%2FrefacNmle3E4%2B6AUn933%2FX6%2F79bnv5P6P%2F3m0tLSsa21trXfzbQMxie3myjAxMfHo1NTUKZVvqjw0Pj6%2B1aoVgxjEckxiP2rV8iDHH8zOzs5lMTMz0zcyMvJBmywb%2BBLD4QLkIJdNlgYF%2BoQcS9kZ6vWfbbZs4OswJSAXOW22OBToWfvOTU9Pt4%2BNjf3Mzeix2jttumTgk%2B0zMYntJh1%2F1qZXx9DQ0Hs1DQP2m9OaeRi5gh23iGAHwngZwMfudPo4MmJbxPIZIHcYZ5B3maK6uvqjFRUVjdTV46LWx4vUFeDXlED67V1dXXVuLgps8XEzjUVsclAnJ7mpZ3EFQTncIuOk%2FkZ9fX0X9cnJyVZ1dIp6Pp%2B%2FfsOGDTdSFyouXry4WaNxj%2FQP8dN6%2BoJG4%2F3W57DFhzoxiEWd2OSgTk5yU78qNPxPK0hA9b0W586dO3eN2v9GrkBzIvCASN0v2StqjyFLQF2jNCzdyxqle2T7lURPDGI5LPn2hkJQ%2FWmLF4bWyvO2nxsdHf2exQEl%2BatVEChmSS0Ek03XNDEuRbsEcljF2n7e4hRXTLF6kZ7u6v2AqwG1%2B11lmuuTpaDYkO9Vlel6Q%2FV%2BZAAb2caaBtkYIJsjmzvBFQQvwyUGhkZiyNUAJCQ7rCR3F4vFW1XfypeiUCjcKtl9SviyTVNcHkMoybEoFPQ5hhso2Q%2Bbm5srtYbe19HRwSfqpFVM1QBr0G4LQhviGyJ10W5M40liEVPqvM7D71vFGnzuktdVIKN9todEtwK2q2TBz1jMmhrSCH3SLotCu%2FzzijFld4jMEFOx%2F6FYnRYj32eXhaFR%2BY7t54UCM7Jft%2FmSoQ494RALgtw2nx8a7ttEII6SLNRLdW66oJJenjh79my1XZaMvr6%2BBmZDMQhGrHRGEpAbDnYJpAtUig%2BtXbv2kHbceyxiQZ%2BU0z6to1d6e3vPNzQ0NGpX9qpkxy4bw8PD1ytm44CwcePGpjVr1nyqqqrqocrKyi024TQ4p3w7a2pqXrMol9MObJDj6%2B5IQFOy%2B%2BjRo2ts8o5hcHCwToR%2BztJJABc42SR22o%2BtizWmKfi2VWWDZnB3lqQ4%2FSQU2u4M%2B6DlHAOLf27eISj3C6bBKPZcuHDhWnbtNy1DWJCQ82lVoEvGNo3iJFwYTXG7F9YtwU7QsKZXqtWClld66xa3X%2BVVfhiFSpR%2FCqsVQjEq%2BLm5Imjk%2Fugq3%2FDbKsRySlu9CoK6WWyvra0tuW3oK%2FBuFV%2FWb1w7%2B6DudmOhMDQtzTou7tbx8xGFieepOjqkeMcV%2Bvfr168%2FFoaGTpiadevW3asqZ%2BlvlC8urAm0xL4o2W%2B5ZChOJ71OMC32Jc%2FKzs7OKiX5A0rWhAg%2BZRWduRkCki9450KHDbZ244vyVOJCbHJYFdBd8U7thdBrDfbgMEwDJ%2B3oO20X6OnpaU6Mgep93d3d1bL7uOrnLeZY%2BpvW7091ED%2FMjzoyq%2FE7jw%2B%2BxLAY%2BRw5nC4gPg9ajf6fDOOrbjNC37VdQF%2BPrQRJoMQHNUU3SNZNW2WXfO5rb2%2B%2F4kBHhg4b23bjSwzagNjksEtA%2Bl9YzQgfRvCk24zEq%2Fp6lAy5kuzS71%2F6HVZvm%2BQUTwISyzd9xGuEGrnKi0Sjvg61FhOfx3qQxJcYxHLMXTYLaISvkW0HtkD2j3OCfyZZE5QK%2BC3bp9izZ0%2FcdLUmrpNNEVvVHwyloKA1%2Bp1QB0%2BrfFMxX7AqgK3jF4mBLImZhQg%2Fjh2Q7ZTibM2dOnWqUoH%2FYnmi%2BKx9SqCRuZ1O6FeQTXqp0E7Oq7f7HQLyJf%2FKwBYffIlhcQm0kT4nfbqexOkg8vyWLVtmFPxHkoWhtneVRjF9w2aR2AhM8YzrOR0lszoedinBiH4dx44de8KqgG3DOROjBMp5u3LHEwQbcXqSegh0IPJYp4pyQufaoWhcBl3HuP0WZdsgm7ssDtTV1U2rYIrP6OgoYYEtPvgSw%2BIScNUjN3W4iNMdodC1hoV5WsqAmKdn3XxQT5%2FBTiPVr6n8tMUB7chqyUous1pXrHFeeazvZyyeF%2BTGDsAJbqyPOxQghCqn9OXYZvt5oXPuBq2P2GmyH1XQX2r97FR5s0Q38aNu2V7ZTGCLD74OMy%2FIDQfs4QQ3dthjCICC%2FL2trW3RS6qOlFtkm33hUXALgQy%2FScsC2OJj9wVBbjjYjc32GAd1uvvU4%2F22XRT9%2Ff21mrJH5M%2B%2FPgY0JWOqj%2FOjjgwdNtjabVHAwXTo2H62d3rd0hTstt2yoKm4Tof0TXpMbeZHHZnVywIcTId%2FvbQw74fdntNR8YjtVg1wMJ05uGk352NrA72u3nqorBKyHOCW1%2Flz3u2c7mmbXV01ZDnAjU%2FUW%2B%2FPXG7bmTNnalwvO5w7PeaCmxbzx7Tj4kyg0MH6JevLDnKbClxm4ZY7cuQIl4W2kAqqn9CVacnHwtsFcpLbNODRBrdQ6qz6asIcqCcl%2F1ktB8jp9DGTcEIel4VCofA7yc9SBzL4gKtlQzYnXODkZlwo709G0OwfsKpsIGeWA5ysit3yUmgEzX1rU1PTf%2FW2XQk2bdpUQW7T4LP7Uii0OK%2FVt7MHIcy1Fr4WilUAuZNRhBPcGFquOHHVVjmmb2H6hi03yA0Hc5mBW15CPi2xWcR6UO%2BLPuqrAXLDwc3glhfLd4lwSFQWDxw4MBKNVQC54UAdTnDL6%2F7PWyJB9Y4dO0rexeWEc6dPhuCmO9f2ZGGqHNX2Tv9fXG6QGw7mwn1we15%2FTquRrDse4Kt2o3HuuKzAaXR09PR%2FABbe2vIkubrTAAAAAElFTkSuQmCC&logoSize=20px&color=blue&link=https%3A%2F%2Fhuggingface.co%2Feurecom-ds%2FPhi-3-mini-4k-socratic)
](https://ollama.com/eurecom-ds/phi-3-mini-4k-socratic)

Using Large Language Models (LLMs) in education presents unique challenges. Typically, LLMs are designed to provide
direct answers to questions, which can hinder students' critical thinking and self-discovery skills. To address this, we
focus on fine-tuning LLMs to facilitate Socratic interactions. Instead of giving straightforward answers, these models
guide students to explore and find the answers themselves. We achieve this through Direct Preference Optimization (DPO).
We test our approach with diverse datasets, including various educational materials and Socratic dialogues. Using
advanced models like GPT-4o for evaluation, our results show that DPO successfully fine-tunes LLMs for Socratic
dialogue, enhancing their educational value.

This repository contains the source material for the paper "Fine Tuning a Large Language Model for Socratic
Interactions" (accepted at KKD-2024).

# Model inference

## HuggingFace

It's possible to download and execute the model using HuggingFace's `transformers` library with:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "eurecom-ds/Phi-3-mini-4k-socratic",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
)

tokenizer = AutoTokenizer.from_pretrained("eurecom-ds/Phi-3-mini-4k-socratic", trust_remote_code=True)
```

Check out for more details at [Phi-3-mini-4k-socratic](https://huggingface.co/eurecom-ds/Phi-3-mini-4k-socratic).

## Ollama

The model is also available at
OllamaHub: [eurecom-ds/phi-3-mini-4k-socratic](https://ollama.com/eurecom-ds/phi-3-mini-4k-socratic). We also made
available the quantized versions for memory constrained environments. Ollama allows swiftly mounting this model in a web
service, or simply for local execution. For example,

```bash
# Ollama installation
curl -fsSL https://ollama.com/install.sh | sh

# Launching ollama service
ollama serve &

# Running the quantized model locally
ollama run eurecom-ds/phi-3-mini-4k-socratic:Q4_0
```

Check out more about Ollama [here](https://github.com/ollama/ollama).

https://github.com/user-attachments/assets/5e7f4b66-332c-48a5-b110-6f5b1a219f39

# Scripts

We also make available evaluation scripts.

- `self_eval.py`: Perform evaluation of the LLM and prompt engineering (e.g., `GPT-4o` or `Llama3:70b`)
- `eval_model.py`: Perform evaluation of the finetuned model or the base model and prompt engineering only
- `gen_train_dataset.py`: Generates the dataset for DPO finetuning using another LLM as a judge (i.e., `GPT-4o`)
- `train.py`: Runs DPO on the base model
- `human_vs_gpt.py`: Use Judge model to perform evaluation of the human scored examples (validation of judge LLM)
- `pipeline.py`: Executes the training pipeline end-to-end (DPO dataset generation + finetuning + evaluation)

For each script, check `--help` for more details.

# Pipeline artifacts

When running the complete pipeline, the script generates a set of training and evaluation artifacts
following the given structure:

```
├── training_root                                  # name to be specified by the user
│   ├── dpo                                        # DPO related files
│   │   ├── {dataset}                              # seed dataset {mathdial,tutorchat,debugging}
│   │       ├── train_dataset.json                 # Examples generated by the base model + prompt engineering then classified in choosen/rejected by the judge model
│   │       ├── weights                            # Finetuned model weights
│   │       ├── checkpoints                        # Training checkpoints
│   ├── evaluation                                 # Performance assements related files
│   │   ├── {dataset}                              # seed dataset {mathdial,tutorchat,debugging}
│   │   │   ├── from_finetuned_with_tutorchat.json # GPT-4o evaluation using model finetuned with tutorchat data 
│   │   │   ├── from_finetuned_with_mathdial.json  #    "        "       "     "       "      "   mathdial data
│   │   │   ├── from_finetuned_with_debugging.json #    "        "       "     "       "      "   debbuging data
│   │   │   ├── base.json                          #    "        "       "     base model + prompt-engineering
│   │   │   ├── gpt4o.json                         #    "        "       "     GPT-4o + prompt-engineering
│   │   ├── human_vs_gpt.json                      # Comparison between human asssessment and judge LLM
│   ├── figures                                    # report evaluation figures
```

# Running in Docker container

It's possible to run any project's script with a Docker container. To do so, first build the image with

```bash
$  docker build -t socratic-llm .
```

Then run it with (tip: don't forget to mount the GPU and script's input/output directories). For example,

```bash
$ docker run --rm --gpus all -v socratic-llm/:/socractic-llm -v /home/<user>/huggingface:/huggingface -e HF_HOME=/huggingface -it socratic-llm -m pipeline --judge-llm openai <open-ai-key> gpt-4o --output-dir /socractic-llm --instruct-model microsoft/Phi-3-mini-4k-instruct
```
