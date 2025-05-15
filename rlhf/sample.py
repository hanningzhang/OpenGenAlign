import os
import argparse
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams


QA_prompt = '''Answer the following question:
{question}
Bear in mind that your response should be strictly based on the following passages:
{passages}
When you response, you should refer to the source of information.'''

DATA_TO_TEXT_prompt = '''Instruction:
Write an objective overview about the following local business based only on the provided structured data in the JSON format.
You should include details and cover the information mentioned in the customers’ review. The overview should be 100 - 200
words. Don’t make up information.
Structured data:
{json_data}
Overview:'''

Summarization_prompt = '''Summarize the following news within {word_num} words:
{news}
output:'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-3.2-3B-Instruct')  # model path
    parser.add_argument("--dataset", type=str, default='../reward/dev_data.json')  # data path
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="data/llama32_sample_n64.json")  # output location
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    dataset = []      
    with open(args.dataset,"r") as f:
       dataset = json.load(f)[:]
    
    gen_dataset = []
    for sample in dataset:
        #prompt = QA_prompt.format(question=sample['question'],passages=sample['reference'])
        gen_dataset.append(sample['chosen'][0])
    
    sampling_params = SamplingParams(n=64, temperature=1.0, top_p=1, max_tokens=2048, stop=[], seed=args.random_seed)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.8,swap_space=32,trust_remote_code=True)

    prompt = gen_dataset
    tokenizer = llm.get_tokenizer()

    format_prompt = []
    for i in prompt:
        conversations = tokenizer.apply_chat_template(
            [i],
            tokenize=False,
        )
        format_prompt.append(conversations)
    #print(prompt[0])
    #format_prompt = llm.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    store_data = []
    completions = llm.generate(format_prompt, sampling_params)
    for i,output in enumerate(completions):
        prompt = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        answers = generated_text
        store_data.append({"prompt":prompt,"answers":answers})
        
    with open(args.output_dir,'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)