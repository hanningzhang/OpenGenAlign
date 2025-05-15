import os
import argparse
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams


QA_prompt = '''Answer the following question and put the final short answer into \\boxed{{}}:
{question}
You should reason step by step to get the final answer. Bear in mind that your response should be strictly based on the following passages:
{passages}
When you respond, you should refer to the source of information.'''

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
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-3.1-8B-Instruct')  # model path
    parser.add_argument("--dataset", type=str, default='data/hotpot_dev_distractor_v1.json')  # data path
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--output_dir", type=str, default="data/llama31_hotpot_dev_n4_reason.json")  # output location
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    dataset = []      
    with open(args.dataset,"r") as f:
       dataset = json.load(f)[:2000]
    
    gen_dataset = []
    for sample in dataset:
        string = ""
        for ct in sample['context']:
            string += ct[0] + '\n\n'
            string += "\n".join(ct[1])
            string += '\n\n'
        
        prompt = QA_prompt.format(question=sample['question'],passages=string)
        gen_dataset.append({"prompt":prompt,"answer":sample['answer'],"reference":string})
    
    sampling_params = SamplingParams(n=4, temperature=1.0, top_p=1, max_tokens=4096, stop=[], seed=args.random_seed)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.8,swap_space=96,trust_remote_code=True)
    
    #prompt_temp = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'
    # prompt_temp = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {input}\nAssistant:"
    # prompt = [prompt_temp.format(input=i['prompt']) for i in gen_dataset]
    # format_prompt = prompt
    prompt = [{"role":"user","content":i['prompt']} for i in gen_dataset]
    tokenizer = llm.get_tokenizer()

    format_prompt = []
    for i in prompt:
        conversations = tokenizer.apply_chat_template(
            [i],
            tokenize=False,
            add_generation_prompt=True
        )
        format_prompt.append(conversations)
    print(format_prompt[0])
    #format_prompt = llm.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    store_data = []
    completions = llm.generate(format_prompt, sampling_params)
    for i,output in enumerate(completions):
        prompt = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        answers = generated_text
        store_data.append({"prompt":prompt,"answers":answers,"ground_truth":gen_dataset[i]['answer'],"reference":gen_dataset[i]['reference']})
        
    with open(args.output_dir,'w') as f:
        json.dump(store_data,f,indent=4,ensure_ascii=False)