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
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List

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

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards
        #return torch.tensor([0])
    
path = "openbmb/UltraRM-13b"
#path = "LxzGordon/URM-LLaMa-3.1-8B"
model = LlamaRewardModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16
).eval().to(0)
tokenizer = LlamaTokenizer.from_pretrained(path)
#tokenizer.model_max_length = 8192
# model.config.pad_token_id = tokenizer.pad_token_id
# tokenizer.truncation_side = "left"
with open("data/llama32_sample_n64.json",'r') as f:
    data = json.load(f)
    
new_data = []
sampling_score = []
for current_sample in tqdm(data[1500:]):
    ls = []
    prompt = current_sample['prompt']
    for ans in current_sample['answers'][:32]:
        prompt, ans = remove_special_tokens(prompt,ans)
        sample1 = "Human: " + prompt + "\nAssistant: " + ans
        inputs = tokenizer(sample1, return_tensors="pt").to(0)
    
        with torch.no_grad():
            score_chosen = model(**inputs).item()
        ls.append(score_chosen)

        #ls.append(1)
    
    # new_ls = [i.item() for i in ls]
    # sampling_score.append(new_ls)
    idx = ls.index(max(ls))
    selected_ans = current_sample['answers'][idx]
    new_data.append({"prompt":prompt,"answer":selected_ans})
    
with open("data/llama32_selected_baseline_v2.json",'w') as f:
    json.dump(new_data,f,indent=4,ensure_ascii=False)

# with open("mistral_sampling_score.json",'w') as f:
#     json.dump(sampling_score,f,indent=4)
