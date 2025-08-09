# Supervised Fine Tuning-LLM using LoRA peft
Fine Tuned LLM to make domain expertise for chart generation.  Used Hermese 2 pro model with Low Rank Adaptation.

NOTE: This model was fine tuned in MPS backend of MacBook. No unlsoth, llama factory, axoloti, etc.. which can be supported on CUDA. 


---
base_model: NousResearch/Hermes-2-Pro-Llama-3-8B
library_name: peft
model_name: lora-finetuned-hermes
tags:
- base_model:adapter:NousResearch/Hermes-2-Pro-Llama-3-8B
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for lora-finetuned-hermes
Hermes maintains its excellent general task and conversation capabilities - but also excels at Function Calling, JSON Structured Outputs, 
and has improved on several other metrics as well, scoring a 90% on our function calling evaluation built in partnership with Fireworks.AI, 
and an 84% on our structured JSON Output evaluation.

This model is a fine-tuned version of [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).
It has been trained using [TRL](https://github.com/huggingface/trl).


## Quick start

```python
from transformers import pipeline

instruction = "Create a table to compare the key performance indicators of Druk School and Ugyen Academy." 
input_data = [  
  {"school": "Druk School", "avg_score": 88, "pass_rate": 95, "attendance": 97},
  {"school": "Ugyen Academy", "avg_score": 91, "pass_rate": 98, "attendance": 96}]


print("Model Response:")
response = generate_response(instruction, input_data)
print(response)
```
Output: {"analysis_text": "Here is a table comparing the key performance indicators of Druk School and Ugyen Academy:", 
"chart": {"type": "table", "data": 
{"columns": 
[{"label": "School", "type": "string"}, 
{"label": "Average Score", "type": "number"}, 
{"label": "Pass Rate", "type": "number"}, 
{"label": "Attendance", "type": "number"}], 
"rows": [{"data": ["Druk School", 88, 95, 97]}, 
{"data": ["Ugyen Academy", 91, 98, 96]}]}}}

## Training procedure
The training steps are included in the jupyter notebook. 
This model was trained with SFT.

## Prompt Format
```python
    start_token = "<|im_start|>assistant"
    end_token = "<|im_end|>"
```
Note: Always use system role format for more accuracy.

References : https://github.com/mlabonne/llm-course?tab=readme-ov-file

## Dataset

[Huggingface: sonamtenzey/instruction_dataset-edu-ai](https://huggingface.co/datasets/sonamtenzey/instruction_dataset-edu-ai)

References : https://github.com/mlabonne/llm-course?tab=readme-ov-file

## Deployment

Merged model (fine-tuned-heremes-2-pro + original model 16GB) were quantized to Q_4_K_M (4.92GB) using llama.cpp server to convert GGUF format. 

References : https://github.com/mlabonne/llm-course?tab=readme-ov-file

## Running the server

``` python 
./build/bin/llama-server \
  -m ../merged-hermes-2-pro-lora-q4_k_m.gguf \
  --port 8080 \
  --host 0.0.0.0 \
  --n-gpu-layers 35 \
  --batch-size 512 \
  --threads 16 \
  -c 2048
```
Or just run python app.py if flask backend is already set up.

## Framework versions
- PEFT 0.17.0
- TRL: 0.21.0
- Transformers: 4.55.0
- Pytorch: 2.3.1
- Datasets: 4.0.0
- Tokenizers: 0.21.4


## Citations
    
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}

@misc{Hermes-2-Pro-Llama-3-8B, 
      url={[https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B]https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)}, 
      title={Hermes-2-Pro-Llama-3-8B}, 
      author={"Teknium", "interstellarninja", "theemozilla", "karan4d", "huemin_art"}
}
