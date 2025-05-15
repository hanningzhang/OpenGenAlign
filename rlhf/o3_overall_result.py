import json
import re
from tqdm import tqdm
import random

with open("data/gpt_overall_o3_temp0_rating_mistral.json",'r') as f:
    data = json.load(f)
    
dataset_webglm = []
invalid = 0
total = 0
count = 0
valid = 0 
count_qa = 0
total_qa = 0
count_data2text = 0
total_data2text = 0
count_summary = 0
total_summary = 0
for i in tqdm(range(len(data))):
    tmp = {}
    prompt = data[i]['prompt']
    answer_1 = data[i]['response_A']
    answer_2 = data[i]['response_B']
    
    rating = data[i]['overall_rating']
    
    A_score = 0
    B_score = 0
    
    is_valid = False
    if "Chosen: A" in rating or "Chosen A" in rating or "Chosen: **A" in rating or "Chosen: Response A" in rating or "**Chosen:** A" in rating or "**Chosen**: **A**" in rating or "**Chosen: (A)**" in rating or "Chosen: **Response A**" in rating:
        A_score += 1
        valid += 1
        is_valid = True
    elif "Chosen: B" in rating or "Chosen B" in rating or "Chosen: **B" in rating or "Chosen: Response B" in rating or "**Chosen:** B" in rating or "**Chosen**: **B**" in rating or "**Chosen: (B)**" in rating or "**Chosen**: B" in rating or "Chosen: (B)" in rating:
        B_score += 1
        valid += 1
        is_valid = True

    if is_valid:
        total += 1
        if "Bear in mind that your response should be strictly based " in prompt:
            total_qa += 1
        if "Write an objective overview about the following" in prompt:
            total_data2text += 1
        if "Summarize the following document" in prompt:
            total_summary += 1
            
            
        if A_score > B_score:
            if "Bear in mind that your response should be strictly based " in prompt:
                count_qa += 1
            if "Write an objective overview about the following" in prompt:
                count_data2text += 1
            if "Summarize the following document" in prompt:
                count_summary += 1
            count += 1

print(valid)    
print(count/total)

print(count_qa/total_qa)
print(total_qa)
print(count_data2text/total_data2text)
print(total_data2text)
print(count_summary/total_summary)
print(total_summary)