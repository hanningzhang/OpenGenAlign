### Instruction

1. go to `/hotpotqa` or `/musique` and `mkdir data`
2. `python gen.py` to generate candidate samples
3. `python rm_select.py` to select the best response
4. `python eval_multiple_rm.py` to evaluate the accuracy for the reward model selected response
5. `python eval_multiple.py` to calculate the average accuracy of multiple responses
