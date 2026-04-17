# LLaRA

- *2024.7*: We have resolved several bugs within our code. Below are the most recent results of LLaRA.

|                | movielens  || steam    || lastfm   ||
|----------------|------------|------|----------|------|----------|------|
|                | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 |
| LLaRA(GRU4Rec) | 0.9684     | 0.4000 | 0.9840 | 0.4916 | 0.9672 | 0.4918 |
| LLaRA(Caser)   | 0.9684     | 0.4211 | 0.9519 | 0.4621 | 0.9754 | 0.4836 |
| LLaRA(SASRec)  | 0.9789     | 0.4526 | 0.9958 | 0.5051 | 0.9754 | 0.5246 |
- *2024.5*: We have updated the Steam dataset to a new version, in which we've addressed an issue that led to the repetition of certain data in the last interacted item of sequence.
- 🔥 *2024.3*: Our paper is accepted by SIGIR'24! Thank all Collaborators! 🎉🎉
- 🔥 *2024.3*: Our [datasets](https://huggingface.co/datasets/joyliao7777/LLaRA) and [checkpoints](https://huggingface.co/joyliao7777/LLaRA) are released on the huggingface.
  
##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/ljy0ustc/LLaRA.git
   cd LLaRA
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).

3. Download the data and checkpoints.

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/` and the checkpoints to the dir path `checkpoints/`.

##### Train LLaRA

Train LLaRA with a single A100 GPU on MovieLens dataset:

```sh
sh train_movielens.sh
```

Train LLaRA with a single A100 GPU on Steam dataset:

```sh
sh train_steam.sh
```

Train LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh train_lastfm.sh
```

Note that: set the `llm_path` argument with your own directory path of the Llama2 model.

##### Evaluate LLaRA

Test LLaRA with a single A100 GPU on MovieLens dataset:

```sh
sh test_movielens.sh
```

Test LLaRA with a single A100 GPU on Steam dataset:

```sh
sh test_steam.sh
```

Test LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh test_lastfm.sh
```

##### Run with `new_data` (Amazon-style split by filename)

- `*_user_items_negs_train.csv` is used for training/validation users.
- `*_user_items_negs_test.csv` is used for test users only (runtime checks enforce no train/test user overlap).
- Backbone LLM can be passed directly as a HuggingFace model id (for example `meta-llama/Llama-2-7b-hf`) and will be downloaded by `from_pretrained`.

Train:
```sh
HF_MODEL_ID=meta-llama/Llama-2-7b-hf DATASET_PREFIX=Baby_Products REC_MODEL_PATH=./rec_model/baby_products.pt bash train_new_data.sh
```

Rank-based inference (1 target + 1000 random negatives):
```sh
HF_MODEL_ID=meta-llama/Llama-2-7b-hf DATASET_PREFIX=Baby_Products REC_MODEL_PATH=./rec_model/baby_products.pt CKPT_PATH=./checkpoints/Baby_Products/last.ckpt bash test_new_data_rank.sh
```

During rank inference, the code prints per-user running averages of HR@10/20/40 and NDCG@10/20/40 after each processed user.

##### Qwen3 native inference (skip training)

If you want to skip training and directly run native Qwen3 inference on test users:

```sh
MODEL_NAME=Qwen/Qwen3-8B DATASET_PREFIX=Baby_Products MAX_USERS=100 bash run_qwen3_native_infer.sh
```

This script:
- loads Qwen3 with `AutoTokenizer` + `AutoModelForCausalLM` from HuggingFace directly;
- samples `1 target + 1000 random negatives` per user by default;
- asks the model to output a full ranking over candidates;
- computes and prints running-average HR@10/20/40 and NDCG@10/20/40 per processed user.
