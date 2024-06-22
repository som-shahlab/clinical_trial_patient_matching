import os
from dotenv import load_dotenv
load_dotenv()
import openai
try:
    import vllm
except ImportError as e:
    pass  # ignore
import pandas as pd
from utils.pipelines import (
    pipeline,
)
from tqdm import tqdm
import argparse
from typing import Dict, List, Any, Optional, Set
from utils.data_loader import XMLDataLoader, ExclusionDataLoader
from utils.types import  UsageStat
from utils.helpers import get_relevant_docs_for_criteria, load_model, load_collection

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key
else:
    raise Exception(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )

PIPELINES = [
    'all_criteria_all_notes',
    'all_criteria_each_notes',
    'each_criteria_all_notes',
    'each_criteria_each_notes',
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test', help='Which split to use')
    parser.add_argument('--embed_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the model to use')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Name of the LLM to use for criteria assessment')
    parser.add_argument('--n_chunks', type=int, default=None, help='# of most similar chunks to use. Use `9999` to return everything')
    parser.add_argument('--threshold', type=float, default=None, help='threshold cutoff to use')
    parser.add_argument('--is_chunk_keep_full_note', action="store_true", default=False, help='If False, then only use specific chunks that meet threshold. If True, then use full note if any chunk within that note meets threshold')
    parser.add_argument('--strategy', type=str, choices=PIPELINES, default=PIPELINES[0], help='What pipeline to use')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Used for vLLM, passed as `tensor_parallel_size` kwarg to vLLM model')
    parser.add_argument('--criterion', type=str, default=None, help='If specfied, then only evaluate that one criterion')
    parser.add_argument('--patient_range', type=str, default=None, help='Format as: `start,end`. If specified, only include patients with those idxs in dataset, inclusive -- e.g. 0,99 will get patients with idxs 0 to 99 inclusive (so 100 total)')
    parser.add_argument('--is_use_orig_defs', action="store_true", default=False, help='If specfied, then use original n2c2 criteria definitions')
    parser.add_argument('--is_exclude_rationale', action="store_true", default=False, help='If specfied, then DO NOT include rationale in JSON output of LLM')
    parser.add_argument('--n_few_shot_examples', type=int, default=None, help='If specfied (int), then provide that number of few shot examples')
    parser.add_argument('--is_exclusion', action="store_true", default=False, help='If specfied, then use custom EXCLUSION criteria')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    llm_model: str = args.llm_model
    embed_model: str = args.embed_model
    path_to_data: str = './data/train/' if args.split == 'train' else './data/n2c2-t1_gold_standard_test_data/test/'
    n_chunks: Optional[int] = args.n_chunks
    threshold: Optional[float] = args.threshold
    is_chunk_keep_full_note: bool = args.is_chunk_keep_full_note
    strategy: str = args.strategy
    criterion: Optional[str] = args.criterion
    is_use_orig_defs: bool = args.is_use_orig_defs
    is_exclusion: bool = args.is_exclusion
    is_exclude_rationale: bool = args.is_exclude_rationale
    n_few_shot_examples: Optional[int] = args.n_few_shot_examples
    tensor_parallel_size: int = args.tensor_parallel_size
    patient_range: Optional[str] = args.patient_range
    date_time = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name: str = f"./outputs/{date_time}{'-exclusion' if is_exclusion else ''}/{llm_model.replace('/', '_')}|{embed_model.replace('/', '_')}|{n_chunks if n_chunks is not None else threshold}|{strategy}|{args.split}{'|chunk' if not is_chunk_keep_full_note else ''}|criteria-{criterion if criterion else 'all'}{'|r_' + str(patient_range) if patient_range else ''}"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # sanity checks
    assert n_chunks is None or threshold is None, "Cannot specify both --n_chunks and --threshold"
    assert n_few_shot_examples is None or n_few_shot_examples == 1 or n_few_shot_examples == 2, f"Currently only supports `1` or `2` or `None` few shot examples, not {n_few_shot_examples}"

    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")
    
    # Load Chroma db
    collection = load_collection(embed_model.split("/")[-1])

    # Load data
    if is_exclusion:
        print("==>", "Using custom EXCLUSION criteria")
        dataloader = ExclusionDataLoader(path_to_data)
    else:
        dataloader = XMLDataLoader(path_to_data)
    dataset = dataloader.load_data()
    
    # Filter by criterion (if specfied)
    if criterion is not None:
        dataloader.criteria = [ criterion ]
        dataloader.definitions = { criterion : dataloader.definitions[criterion] }
    
    # Adjust definitions (if using original)
    if is_use_orig_defs:
        dataloader.definitions = dataloader.original_definitions
    
    # Select patient range (if specfied)
    if patient_range is not None:
        start_idx, end_idx = [ int(x) for x in patient_range.split(',') ]
        assert start_idx <= end_idx, "Start idx must be <= end idx"
        assert start_idx >= 0 and end_idx < len(dataset), f"Start idx must be >= 0 and end idx must be < {len(dataset)}"
        dataset = dataset[int(start_idx):int(end_idx)+1]

    # Load embed model
    model, tokenizer = load_model(embed_model)
    
    # Load LLMs
    llm_kwargs = {}
    if 'gpt-' in llm_model or 'GPT4' in llm_model:
        # OpenAI model
        if llm_model in ['GPT4-32k', 'GPT4-32k-2', 'shc-gpt-35-turbo-16k']:
            # Azure SHC
            llm_kwargs['openai_client'] = openai.AzureOpenAI(
                base_url=f"https://shcopenaisandbox.openai.azure.com/openai/deployments/{llm_model}/chat/completions?api-version=2024-02-15-preview",
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
            )
            llm_kwargs['is_use_json'] = False
        else:
            llm_kwargs['openai_client'] = openai.OpenAI()
            llm_kwargs['is_use_json'] = True
    else:
        # HF model using Outlines and vLLM
        llm_kwargs['model'] = vllm.LLM(model=llm_model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        llm_kwargs['tokenizer'] = llm_kwargs['model'].get_tokenizer()

    # For each patient...
    results: List[Dict[str, Any]] = []
    stats: List[UsageStat] = []
    patient_2_criteria_2_docs: Dict[str, Dict[str, List]] = {}
    
    for patient_idx, patient in enumerate(tqdm(dataset, desc='Looping through patients...')):
        # Query db for relevant docs
        criterion_2_docs: Dict[str, List[Dict[str, Any]]] = get_relevant_docs_for_criteria(dataloader, 
                                                                                            collection, 
                                                                                            patient['patient_id'], 
                                                                                            model, 
                                                                                            tokenizer, 
                                                                                            n_chunks=n_chunks,
                                                                                            threshold=threshold,
                                                                                            is_all_criteria='all_criteria' in strategy)
        patient_2_criteria_2_docs[patient['patient_id']] = criterion_2_docs
    print(f"# of patients: {len(patient_2_criteria_2_docs)}")
    print(f"# of notes: {sum([ len(patient_2_criteria_2_docs[patient][criterion]) for patient in patient_2_criteria_2_docs for criterion in patient_2_criteria_2_docs[patient] ])}")

    # Query LLM for all patients
    results, stats = pipeline(dataloader, 
                              dataset, 
                              patient_2_criteria_2_docs, 
                              llm_model, 
                              llm_kwargs, 
                              is_all_criteria='all_criteria' in strategy, 
                              is_all_notes='all_notes' in strategy,
                              is_chunk_keep_full_note=is_chunk_keep_full_note,
                              is_exclude_rationale=is_exclude_rationale,
                              n_few_shot_examples=n_few_shot_examples)


    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{file_name}.csv")
    print(f"Result Saved at {file_name}")

    # Calculate overall token usage
    completion_tokens: int = sum([ stat.completion_tokens for stat in stats ])
    prompt_tokens: int = sum([ stat.prompt_tokens for stat in stats ])
    # Calculate cost
    cost_per_1k_completion_tokens: float = 0.0
    cost_per_1k_prompt_tokens: float = 0.0
    if llm_model == 'gpt-4-1106-preview':
        cost_per_1k_completion_tokens = 0.01
        cost_per_1k_prompt_tokens = 0.03
    elif llm_model == 'gpt-4-32k' or llm_model == 'GPT4-32k' or llm_model == 'GPT4-32k-2':
        cost_per_1k_completion_tokens = 0.06
        cost_per_1k_prompt_tokens = 0.12
    elif llm_model == 'gpt-4':
        cost_per_1k_completion_tokens = 0.03
        cost_per_1k_prompt_tokens = 0.06
    elif llm_model in ['gpt-3.5-turbo-1106', 'shc-gpt-35-turbo-16k']:
        cost_per_1k_completion_tokens = 0.0015
        cost_per_1k_prompt_tokens = 0.0020

    with open(f"{file_name}_usage.txt", 'w') as f:
        f.write(f"# of patients: {len(dataset)}\n")
        f.write(f"# of completion tokens: {completion_tokens}\n")
        f.write(f"# of prompt tokens: {prompt_tokens}\n")
        f.write(f"Total # of tokens: {completion_tokens + prompt_tokens}\n")
        f.write(f"Cost: ${(completion_tokens * cost_per_1k_completion_tokens / 1000 + prompt_tokens * cost_per_1k_prompt_tokens / 1000 )}\n")
    
