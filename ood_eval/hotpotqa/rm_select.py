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

QA_prompt = '''Answer the following question:
{question}
Bear in mind that your response should be strictly based on the following passages:
{passages}
When you response, you should always cite the source of information.'''

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
    
path = "HanningZhang/Llama3.1-RAG-Reward"
#path = "nicolinho/QRM-Llama3.1-8B-v2"
model = AutoModelForSequenceClassification.from_pretrained(
    path, 
    num_labels=1, 
    torch_dtype=torch.bfloat16, 
    #use_flash_attention_2=True,
).eval().to(0)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast = False)
#tokenizer.model_max_length = 8192
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.truncation_side = "left"
with open("data/llama31_hotpot_dev_n4_reason.json",'r') as f:
    data = json.load(f)

count = 0
total = 0

for sample in tqdm(data):
    prompt = sample['prompt']
    prompt = prompt.replace("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n","")
    prompt = prompt.replace("<|im_end|>\n<|im_start|>assistant\n","")
    prompt = prompt.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n","")
    prompt = prompt.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","")
    score_list = []
    for ans in sample['answers'][:4]:
        tmp1 = [{"role":"user","content":prompt},
                {"role":"assistant","content":ans}]
        sample1 = {}
        sample1['positive'] = tokenizer.apply_chat_template(
            tmp1, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        tokenized_pos2 = tokenizer(sample1['positive'], truncation=True,return_tensors='pt').to(0)
        sample1["input_ids"] = tokenized_pos2["input_ids"]
        sample1["attention_mask"] = tokenized_pos2["attention_mask"]
        score_chosen = compute_loss(model,sample1)
        score_list.append(score_chosen)
        
    idx = score_list.index(max(score_list))
    sample['max_idx'] = idx
    
with open("data/llama31_hotpot_dev_n4_reason_reward_select.json",'w') as f:
    json.dump(data,f,indent=4,ensure_ascii=False)

