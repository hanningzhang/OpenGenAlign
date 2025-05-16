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
    
path = "llama3_rm_all/last_checkpoint"
#path = "nicolinho/QRM-Llama3.1-8B-v2"
model = AutoModelForSequenceClassification.from_pretrained(
    path, 
    num_labels=1, 
    torch_dtype=torch.bfloat16, use_flash_attention_2=True,
).eval().to(0)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast = False)
#tokenizer.model_max_length = 8192
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.truncation_side = "left"
with open("test_data_o3.json",'r') as f:
    data = json.load(f)

count = 0
total = 0
count_qa = 0
count_data2text = 0
count_summary = 0
length_correct = []
length_incorrect = []
for current_sample in tqdm(data):
    tmp1 = current_sample['chosen']
    sample1 = {}
    sample1['positive'] = tokenizer.apply_chat_template(
            tmp1, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    tokenized_pos2 = tokenizer(sample1['positive'], truncation=True,return_tensors='pt').to(0)
    sample1["input_ids"] = tokenized_pos2["input_ids"]
    sample1["attention_mask"] = tokenized_pos2["attention_mask"]
    score_chosen = compute_loss(model,sample1)
        
    tmp2 = current_sample['rejected']
    #tmp2 = [{"role":"user","content":"test"},{"role":"assistant","content":"test"}]
    sample2 = {}
    sample2['positive'] = tokenizer.apply_chat_template(
        tmp2, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
    tokenized_pos2 = tokenizer(sample2['positive'], truncation=True,return_tensors='pt').to(0)
    sample2["input_ids"] = tokenized_pos2["input_ids"]
    sample2["attention_mask"] = tokenized_pos2["attention_mask"]
    score_rejected = compute_loss(model,sample2)
    
    total += 1
    # print(score_chosen)
    # print(score_rejected)
    if score_chosen > score_rejected:
        if "Bear in mind that your response should be strictly based " in tmp1[0]['content']:
            count_qa += 1
        if "Write an objective overview about the following" in tmp1[0]['content']:
            count_data2text += 1
        if "Summarize the following document" in tmp1[0]['content']:
            count_summary += 1
        count += 1
        
print(count/total)

print(count_qa/500)
print(count_data2text/500)
print(count_summary/500)
# print(sum(length_correct)/len(length_correct))
# print(sum(length_incorrect)/len(length_incorrect))
#print(sum(score_list)/len(score_list))
# with open("data/hallucination_selected.json",'w') as f:
#     json.dump(dataset_hallucination,f,indent=4)

