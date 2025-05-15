from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
# import evaluate
import numpy as np
import json
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

def remove_special_tokens(prompt,answer):
    
    prompt = prompt.replace("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n","")
    prompt = prompt.replace("<|eot_id|>","")
    prompt = prompt.replace("<s> [INST] ","")
    prompt = prompt.replace(" [/INST]","")
    
    answer = answer.replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")
    answer = answer.replace("assistant\n","")
    answer = answer.replace("user\n","")
    answer = answer.replace("Assistant:","")
    answer = answer.replace("answer\n","")
    answer = answer.replace("user\n","")
    answer = answer.replace("document\n","")

    return prompt, answer.strip()

def compute_loss(model, inputs, return_outputs=False):
    with torch.no_grad():
        #print(inputs['input_ids'].shape)
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        return rewards[0]
        # rewards = model(
        #     input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        # ).logits[0][0]
        # return rewards
    
path = "../reward/llama3_o3_rm/last_checkpoint"
#path = "LxzGordon/URM-LLaMa-3.1-8B"
model = AutoModelForSequenceClassification.from_pretrained(
    path, 
    num_labels=1, 
    torch_dtype=torch.bfloat16, use_flash_attention_2=True,
).eval().to(0)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast = False)
#tokenizer.model_max_length = 8192
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.truncation_side = "left"
with open("data/mistral1_dev.json",'r') as f:
    data = json.load(f)
    
new_data = []
sampling_score = []
for current_sample in tqdm(data[:50]):
    ls = []
    prompt = current_sample['prompt']
    for ans in current_sample['answers']:
        prompt, ans = remove_special_tokens(prompt,ans)
        tmp1 = [
            {"role":"user","content":prompt},
            {"role":"assistant","content":ans}
        ]
        sample1 = {}
        sample1['positive'] = tokenizer.apply_chat_template(
            tmp1, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        tokenized_pos2 = tokenizer(sample1['positive'], truncation=True,return_tensors='pt').to(0)
        sample1["input_ids"] = tokenized_pos2["input_ids"]
        sample1["attention_mask"] = tokenized_pos2["attention_mask"]
        score_chosen = compute_loss(model,sample1)
        ls.append(score_chosen)
    
    new_ls = [i.item() for i in ls]
    sampling_score.append(new_ls)
    idx = ls.index(max(ls))
    selected_ans = current_sample['answers'][idx]
    new_data.append({"prompt":prompt,"answer":selected_ans})
    
# with open("data/llama3_selected_o3_v2.json",'w') as f:
#     json.dump(new_data,f,indent=4,ensure_ascii=False)

with open("mistral_sampling_score.json",'w') as f:
    json.dump(sampling_score,f,indent=4)
