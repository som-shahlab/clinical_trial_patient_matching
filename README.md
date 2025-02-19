# Installation

How to install the repo:

```bash
conda create -n pmct python=3.10 -y
conda activate pmct
pip install -r requirements.txt
pip install -e .
```

To download the n2c2 2018 cohort challenge dataset used in this paper, [please sign up for dataset access here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)

# Generate Embeddings

For our retrieval pipeline experiments, we need to first generate embeddings for each patients' notes. We test the `BAAI/bge-large-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2` models:

```bash
# Train
python scripts/create_db.py --path_to_data ./data/train/ --embed_model 'BAAI/bge-large-en-v1.5'
python scripts/create_db.py --path_to_data ./data/train/ --embed_model 'sentence-transformers/all-MiniLM-L6-v2'
# Test
python scripts/create_db.py --path_to_data ./data/n2c2-t1_gold_standard_test_data/test/ --embed_model 'BAAI/bge-large-en-v1.5'
python scripts/create_db.py --path_to_data ./data/n2c2-t1_gold_standard_test_data/test/ --embed_model 'sentence-transformers/all-MiniLM-L6-v2'
```

# Experiments

## Prompt Strategy

Compare the "ACIN", "ACAN", "ICAN", and "ICIN" strategies for both GPT-4 and GPT-3.5.

```bash
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999

python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
```

## Prompt Engineering

Compare the "original" definitions to the "improved" definitions.

```bash
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999 --is_use_orig_defs
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999 --is_use_orig_defs
```


## Retrieval Pipeline

Test different pre-filtering top-$k$ values..

Note: When using `each_criteria_*_notes`, each criterion gets looked up individually, so `n_chunks=1` will really return 13 chunks (one for each criterion). When using `all_criteria_*_notes`, all criteria are combined into one string and looked up at once, so `n_chunks=1` will really return 1 chunk (one for all criteria).

```bash
# GPT-4
# BAAI/bge-large-en-v1.5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
# sentence-transformers/all-MiniLM-L6-v2
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10

# GPT-3.5
# BAAI/bge-large-en-v1.5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'BAAI/bge-large-en-v1.5' --n_chunks 10
# sentence-transformers/all-MiniLM-L6-v2
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 1
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 5
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --embed_model 'sentence-transformers/all-MiniLM-L6-v2' --n_chunks 10
```

## LLM Models

Test different open source models.

```bash
python scripts/eval.py --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --llm_model 'mistralai/Mixtral-8x7B-Instruct-v0.1' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999 --tensor_parallel_size 4
python scripts/eval.py --llm_model 'NousResearch/Yarn-Llama-2-70b-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999 --tensor_parallel_size 4
# TBD
python scripts/eval.py --llm_model 'Qwen/Qwen-72B-Chat' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999 --tensor_parallel_size 4
```

## Ablation of `Rationale` key in prompt

```bash
python scripts/eval.py --is_exclude_rationale --llm_model 'GPT4-32k' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'GPT4-32k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'GPT4-32k' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'GPT4-32k-2' --strategy 'each_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999

python scripts/eval.py --is_exclude_rationale --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --is_exclude_rationale --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
```

## Few-Shot

```bash
# One shot
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'GPT4-32k-2' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'GPT4-32k-2' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'GPT4-32k-2' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'GPT4-32k-2' --strategy 'each_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999

python scripts/eval.py --n_few_shot_examples 1 --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_each_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'shc-gpt-35-turbo-16k' --strategy 'all_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
python scripts/eval.py --n_few_shot_examples 1 --llm_model 'shc-gpt-35-turbo-16k' --strategy 'each_criteria_all_notes' --is_chunk_keep_full_note --n_chunks 9999
```

## Troubleshooting

Sometimes the models will not generate valid JSON outputs. If this is the case, run the following script to identify problematic outputs and requeue them. You will need to point the `PATH_TO_RESULTS_CSV` variable to the location of the results CSV file you want to generate metrics for.

```bash
python scripts/post_hoc_cleanup.py PATH_TO_RESULTS_CSV
```

# Generate Results

Calculate metrics by running the following script. You will need to point the `PATH_TO_RESULTS_CSV` variable to the location of the results CSV file you want to generate metrics for.

```bash
python scripts/gen_metrics.py PATH_TO_RESULTS_CSV
```

# Generate Figures

To generate figures, run the following script.

```bash
python scripts/make_figures.py
python scripts/make_clinician_rationale_figure.py
```

# Citation

If you found this work helpful, please consider citing us!

```
@article{wornow2025zero,
  title={Zero-shot clinical trial patient matching with llms},
  author={Wornow, Michael and Lozano, Alejandro and Dash, Dev and Jindal, Jenelle and Mahaffey, Kenneth W and Shah, Nigam H},
  journal={NEJM AI},
  volume={2},
  number={1},
  pages={AIcs2400360},
  year={2025},
  publisher={Massachusetts Medical Society}
}
```
