## Environment

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets

# The following code is tested for CUDA12.0-12.2, and CUDA12.6
# To develop llama-3, mistral, gemma-1, 1.1, 2, deepseek you can consider the following vllm version
pip install vllm==0.5.4

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.43.4
pip install numpy==1.26.4
```

## Instruction

1. go to `/hotpotqa` or `/musique` and `mkdir data`
2. `python gen.py` to generate candidate samples
3. `python rm_select.py` to select the best response
4. `python eval_multiple_rm.py` to evaluate the accuracy for the reward model selected response
5. `python eval_multiple.py` to calculate the average accuracy of multiple responses
