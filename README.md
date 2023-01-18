# E2E-SLU
Code for our paper, ""

Abstract:

Notes: 
Submitted to ~

## Contributions
1. We "open source" our MOS (MultiOz-Speech) dataset
2. E2E
3. Provide baseline model
4. (hopefully) Results are similar to other well known models (near SOTA)

## Dataset
1. Used LJSpeech dataset to train modified Tacotron2 (used SMA instead of LSA)
2. The audio samples are of 16000Hz, 16 bits, mono
3. The dataset is organized in terms of dialogue
    - User, system turns for a particular dialogue is placed in the same folder
        - 8424 dialogues in train
        - 1000 dialogues in test
        - 1000 dialogues in dev 


