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

with open("data/mistral1_test_beforerlhf.json",'r') as f:
    data1 = json.load(f)
    
with open("data/mistral1_afterrlhf_o3_baseline.json",'r') as f:
    data2 = json.load(f)
   
count = 0
total = 0 
count_qa = 0
count_data2text = 0
count_summary = 0
qa = []
yelp = []
xsum = []
for i,current_sample in enumerate(tqdm(data1)):
    if "Bear in mind that your response should be strictly based " in current_sample['prompt']:
        prompt = current_sample['prompt']
        ans1 = current_sample['answers'][0]
        ans2 = data2[i]['answers'][0]
        prompt, ans1 = remove_special_tokens(prompt,ans1)
        prompt, ans2 = remove_special_tokens(prompt,ans2)
        tmp = {
            "prompt":prompt,
            "response_A":ans1,
            "response_B":ans2
        }
        qa.append(tmp)
    if "Write an objective overview about the following" in current_sample['prompt']:
        prompt = current_sample['prompt']
        ans1 = current_sample['answers'][0]
        ans2 = data2[i]['answers'][0]
        prompt, ans1 = remove_special_tokens(prompt,ans1)
        prompt, ans2 = remove_special_tokens(prompt,ans2)
        tmp = {
            "prompt":prompt,
            "response_A":ans1,
            "response_B":ans2
        }
        yelp.append(tmp)
    if "Summarize the following document" in current_sample['prompt']:
        prompt = current_sample['prompt']
        ans1 = current_sample['answers'][0]
        ans2 = data2[i]['answers'][0]
        prompt, ans1 = remove_special_tokens(prompt,ans1)
        prompt, ans2 = remove_special_tokens(prompt,ans2)
        tmp = {
            "prompt":prompt,
            "response_A":ans1,
            "response_B":ans2
        }
        xsum.append(tmp)

with open("human_webglm.json",'w') as f:
    json.dump(qa,f,indent=4,ensure_ascii=False)
    
with open("human_yelp.json",'w') as f:
    json.dump(yelp,f,indent=4,ensure_ascii=False)
    
with open("human_xsum.json",'w') as f:
    json.dump(xsum,f,indent=4,ensure_ascii=False)
    
    
total_score_before = 0
total_score_after = 0
for i,current_sample in enumerate(tqdm(data1)):
    ls = []
    prompt = current_sample['prompt']
    ans = current_sample['answers'][0]
    
    prompt, ans = remove_special_tokens(prompt,ans)
    tmp_before = [
            {"role":"user","content":prompt},
            {"role":"assistant","content":ans}
    ]
    sample1 = {}
    sample1['positive'] = tokenizer.apply_chat_template(
        tmp_before, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    tokenized_pos2 = tokenizer(sample1['positive'], truncation=True,return_tensors='pt').to(0)
    sample1["input_ids"] = tokenized_pos2["input_ids"]
    sample1["attention_mask"] = tokenized_pos2["attention_mask"]
    score_before = compute_loss(model,sample1)
    
    ans = data2[i]['answers'][0]
    prompt, ans = remove_special_tokens(prompt,ans)
    tmp_after = [
            {"role":"user","content":prompt},
            {"role":"assistant","content":ans}
    ]
    sample1 = {}
    sample1['positive'] = tokenizer.apply_chat_template(
        tmp_after, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    tokenized_pos2 = tokenizer(sample1['positive'], truncation=True,return_tensors='pt').to(0)
    sample1["input_ids"] = tokenized_pos2["input_ids"]
    sample1["attention_mask"] = tokenized_pos2["attention_mask"]
    score_after = compute_loss(model,sample1)
    
    total += 1
    
    total_score_after += score_after
    total_score_before += score_before
    
    if score_after > score_before:
        count += 1
        if "Bear in mind that your response should be strictly based " in tmp_before[0]['content']:
            count_qa += 1
        if "Write an objective overview about the following" in tmp_before[0]['content']:
            count_data2text += 1
        if "Summarize the following document" in tmp_before[0]['content']:
            count_summary += 1
        
    print(count/total)

print(total_score_after/1500)
print(total_score_before/1500)
print(count_qa/500)
print(count_data2text/500)
print(count_summary/500)
    # idx = ls.index(max(ls))
    # selected_ans = current_sample['answers'][idx]
    # new_data.append({"prompt":prompt,"answer":selected_ans})
    
# with open("data/mistral1_selected.json",'w') as f:
#     json.dump(new_data,f,indent=4,ensure_ascii=False)

