## Download the data
Download the following files in the folder `https://drive.google.com/drive/folders/18th8lS1TOoQGZDfb1EvapVRvMstnVRUv?usp=sharing` and put into current directory.

## Environments

```shell
conda create -n rm_dev python=3.10.9
conda activate rm_dev

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .

pip install flash-attn==2.6.3

pip install accelerate==0.33.0 # for gemma2 and llama3.1
pip install deepspeed==0.12.2
pip install transformers==4.43.4
pip install numpy==1.26.4 # Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!

pip install wandb
```

## Instructions

training the reward model
```
accelerate launch llama3_rm.py --model_name {Your Model} --max_length 4096 --train_set_path {JSON files} --deepspeed ./deepspeed_configs/deepspeed_3.json
```
`train_data_o3.json` is the OpenGenAlign dataset
`train_rlhflow.json` is the Baseline dataset in Section 6
`train_all.json` is the MixReward dataset in Section 6

## Evaluation

`python eval.py` to evaluate on the OpenGenAlign Benchmark
`python eval_rewardbench.py` to evaluate on the RewardBench

