## Environment

```shell
conda create -n sft python=3.10.9
conda activate sft

pip install vllm==0.5.4
## Get axolotl for general model
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

# The test cuda version is 12.1, 12.2. You may need to update the torch version based on your cuda version...
# you may encounter underfined symbol error related to cuda and flash-attn and 2.1.2 can solve it ...
pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn

# fix an error of axolotl: ModuleNotFoundError: No module named 'pynvml.nvml'; 'pynvml' is not a package
pip install nvidia-ml-py3
# also edit axolotl/src/axolotl/utils/bench.py (line 6) to: ``from pynvml import NVMLError''


## Get FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

pip install deepspeed
```

## Instruction

1. `mkdir data` and `python sample.py` to sample multiple responses
2. `python select_data.py` to use reward model to select the data
3. `axolotl train` to train on the selected data
4. `python test_sample` to sample response using the intial model and the post-rlhf model
5. `python eval.py` to use reward model to compare the response from Step 4
6. `python o3_overall_eval.py` and `o3_overall_result` to use o3 to compare the response from Step 4

**You might need to modify the file path, model path inside the files before running**
