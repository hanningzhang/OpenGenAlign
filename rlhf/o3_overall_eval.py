import os
import argparse
from tqdm import tqdm
import json
import random
import copy
from openai import OpenAI

comprehensive_system_prompt = '''You are an expert in evaluating the comprehensiveness of Retrieval-Augmented Generation outputs. 
You will compare two responses, Response A and Response B, assessing which one provides a more comprehensive answer based on the reference. 
Consider the depth, coverage, and completeness of key points. 
Provide a short (within 100 words) explanation for your judgment, and identify which one (A or B) is the better one (format: Chosen: (A or B)).
'''

hallucination_system_prompt = '''You are a expert in evaluating the accuracy of Retrieval-Augmented Generation outputs, focusing specifically on hallucinations. 
Compare two responses, Response A and Response B, and determine which one is more accurate and reliable based on the reference.
Provide a short (within 100 words) explanation for your judgment, and identify which one (A or B) is the better one (format: Chosen: (A or B)).
'''

efficiency_system_prompt = '''You are an expert in evaluating the verbosity of Retrieval-Augmented Generation outputs, focusing on conciseness, clarity, and relevance. 
You will compare two responses, Response A and Response B, assessing which one more efficiently answers the question.
Provide a short (within 100 words) explanation for your judgment, and identify which one (A or B) is the better one (format: Chosen: (A or B)).
'''

citation_system_prompt = '''You are an expert in evaluating Retrieval-Augmented Generation (RAG) outputs. Your task is to compare two responses, Response A and Response B, based on the following two aspects:
Generation Accuracy and Relevance: Does the response directly and accurately answer the question? Is the information presented relevant and free from hallucinations or unsupported claims?
Quality of References and Contextualization: Does the response effectively incorporate and reference the retrieved content or context? How well does the response provide contextual information or retrieved details to enhance the user's understanding?
Provide a short (within 100 words) explanation for your judgment, and identify which one (A or B) is the better one (format: Chosen: (A or B)).
'''

with_citation_system_prompt = '''You are an expert in evaluating question-answering (QA) responses in a Retrieval-Augmented Generation (RAG) setting. Your task is to compare two responses, Response A and Response B, based on the following criteria, with a stronger emphasis on faithfulness and comprehensiveness:

Primary Criteria (Most Important)
Faithfulness (Hallucination & Accuracy) – Highest Priority

Assess whether each response strictly adheres to the provided reference passages.
A good response must not introduce fabricated, misleading, or unverifiable information—it should only contain details supported by the reference.
Identify which response is more factually accurate and better grounded in the reference material.

Comprehensiveness (Coverage & Relevance) – Second Priority

Determine how well each response fully answers the given question, ensuring that all essential aspects are covered.
A strong response should capture key details from the reference while avoiding unnecessary or irrelevant information.
Identify which response provides a more complete and well-supported answer.

Secondary Criteria (Supporting Factors)
Conciseness (Efficiency & Clarity)

Evaluate how effectively each response conveys the necessary information without excessive verbosity.
The ideal response should be succinct yet informative, avoiding redundant details while preserving key insights.
If two responses are equally faithful and comprehensive, prefer the one that is more concise and well-structured.

Citation (Attribution & Use of Retrieved Content)

Examine how well each response incorporates and attributes information from the retrieved passages.
A strong response should clearly reference relevant sources when necessary, ensuring that retrieved content supports the answer.
If two responses are otherwise equal, prefer the one that makes better use of the retrieval sources.

Final Judgment:
Focus primarily on faithfulness and comprehensiveness when deciding which response is superior. Provide a concise explanation of your reasoning, then explicitly state which response is better using the following format:

Chosen: (A or B)
'''

xsum_system_prompt = '''You are an expert in evaluating the quality of summarization in a Retrieval-Augmented Generation (RAG) setting. Your task is to compare two summaries, Response A and Response B, based on the following criteria:

Faithfulness (Hallucination & Accuracy):

Assess how well each summary adheres strictly to the provided reference.
A good summary should only include verifiable information from the reference and avoid adding any fabricated, misleading, or exaggerated details.
Identify which summary is more factually accurate and better grounded in the reference.

Coverage (Comprehensiveness & Relevance):

Determine how well each summary captures the key points of the reference without omitting essential details.
A strong summary should convey all critical aspects of the original content while avoiding irrelevant or unnecessary information.
Identify which summary provides a more complete and well-balanced representation of the reference.

Conciseness (Efficiency & Clarity):

Compare how effectively each summary delivers the key information in a compact and clear manner.
An ideal summary should be succinct yet informative, avoiding excessive verbosity while retaining all necessary details.
Determine which summary is more precise and effectively worded.
Final Judgment:
Based on the above criteria, provide a brief explanation of your decision. Then, explicitly state which summary is the better one in the following format:

Chosen: (A or B)
'''

yelp_system_prompt = '''You are an expert in evaluating Retrieval-Augmented Generation (RAG) outputs. Your task is to compare two responses, Response A and Response B, based on three key evaluation criteria:
1. Hallucination: Assess the accuracy and reliability of the responses. A response should strictly adhere to the provided reference and contain no fabricated or misleading information. Determine which response is more grounded in the reference.
2. Comprehensiveness: Evaluate the depth and coverage of the responses. A comprehensive response should address all key aspects of the question while integrating relevant details from the reference. Identify which response provides a more complete and well-rounded answer.
3. Verbosity: Compare the efficiency of the responses in answering the question. An ideal response should be clear, concise, and free of unnecessary elaboration while maintaining informativeness. Determine which response presents the information in a more precise and effective manner.
Provide a short explanation, and in the end identify which one (A or B) is the better one (format: Chosen: (A or B)).
'''

user_prompt = '''
Question and reference: 
{Question}
Response A: 
{Answer_A}
Response B:
{Answer_B}
Output:
'''
client = OpenAI(
    # This is the default and can be omitted
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=""
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
    
def gpt_response(system_message,content):
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "system", 
            "content": system_message
        },
        {
            "role": "user",
            "content": content,
        }
        ],
        model="o3-mini",
        reasoning_effort="medium",
        #temperature=0.0
    )
    return chat_completion.choices[0].message.content

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_1", type=str, default='data/mistral1_afterrlhf_o3_baseline.json')  # data path
    parser.add_argument("--dataset_2", type=str, default='data/mistral1_test_beforerlhf.json')  # data path
    parser.add_argument("--output_dir", type=str, default="data/gpt_overall_o3_mistral_baseline.json")  # output location
    parser.add_argument("--random_seed", type=int, default=42)  # random seed
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #random.seed(42)
    with open(args.dataset_1,'r') as f:
        post_data = json.load(f)
        
    with open(args.dataset_2,'r') as f:
        data = json.load(f)
    
    new_data = []
    for i,sample in enumerate(tqdm(data)):
        
        prompt = post_data[i]['prompt']
        post_answer = post_data[i]['answers'][0]
        answer = sample['answers'][0]

        prompt, post_answer = remove_special_tokens(prompt,post_answer)
        prompt, answer = remove_special_tokens(prompt,answer)
        
        new_sample = {"prompt":prompt, "response_A":post_answer,"response_B":answer}
        
        sample_prompt = user_prompt.format(Question=prompt,Answer_A=post_answer,Answer_B=answer)
        
        if "Bear in mind that your response should be strictly based" in prompt:
            citation_rating = gpt_response(with_citation_system_prompt,sample_prompt)
            new_sample['overall_rating'] = citation_rating
        elif "Write an objective overview about the" in prompt:
            citation_rating = gpt_response(yelp_system_prompt,sample_prompt)
            new_sample['overall_rating'] = citation_rating
        else:
            citation_rating = gpt_response(xsum_system_prompt,sample_prompt)
            new_sample['overall_rating'] = citation_rating
 
        new_data.append(new_sample)

        with open(args.output_dir,'w') as f:
            json.dump(new_data,f,indent=4,ensure_ascii=False)   