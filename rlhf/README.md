### Instruction

1. `mkdir data` and `python sample.py` to sample multiple responses
2. `python select_data.py` to use reward model to select the data
3. `axolotl train` to train on the selected data
4. `python test_sample` to sample response using the intial model and the post-rlhf model
5. `python eval.py` to use reward model to compare the response from Step 4
6. `python o3_overall_eval.py` and `o3_overall_result` to use o3 to compare the response from Step 4
