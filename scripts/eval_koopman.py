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
    pipeline_koopman,
)
from tqdm import tqdm
import argparse
from typing import Dict, List, Any, Optional, Set
from utils.data_loader import XMLDataLoaderKoopman
from utils.types import  UsageStat

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
    parser.add_argument('--llm_model',             type=str, default='gpt-3.5-turbo-0125', help='Name of the LLM')
    parser.add_argument('--tensor_parallel_size',  type=int, default=1,   help='Passed as `tensor_parallel_size` kwarg to vLLM model')
    parser.add_argument('--patient_range', type=str, default=None, help='Format as: `start,end`. If specified, only include patients with those idxs in dataset, inclusive -- e.g. 0,99 will get patients with idxs 0 to 99 inclusive (so 100 total)')
    parser.add_argument('--is_exclude_rationale', action="store_true", default=False, help='If specfied, then DO NOT include rationale in JSON output of LLM')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args                          = parse_args()
    llm_model: str                = args.llm_model
    is_exclude_rationale: bool    = args.is_exclude_rationale
    tensor_parallel_size: int     = args.tensor_parallel_size
    patient_range: Optional[str]  = args.patient_range
    date_time                     = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name: str                = f"./outputs/{date_time}/koopman_{llm_model.replace('/', '_')}"
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")
    
    

    # Load data
    dataloader = XMLDataLoaderKoopman("/Users/michaelwornow/Desktop/clinical_trial_patient_matching/data/koopman/data/")
    dataset    = dataloader.load_data(load_from_json = "/Users/michaelwornow/Desktop/clinical_trial_patient_matching/data/patient_trail_dataset.json")
    
    # Select patient range (if specfied)
    if patient_range is not None:
        start_idx, end_idx = [ int(x) for x in patient_range.split(',') ]
        assert start_idx <= end_idx, "Start idx must be <= end idx"
        assert start_idx >= 0 and end_idx < len(dataset["patients"]), f"Start idx must be >= 0 and end idx must be < {len(dataset)}"
        dataset['patients'] = dataset["patients"][int(start_idx):int(end_idx)+1]
    
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
            llm_kwargs['openai_client'] = openai.OpenAI(api_key=api_key)
            llm_kwargs['is_use_json'] = True
    else:
        # HF model using Outlines and vLLM
        llm_kwargs['model']     = vllm.LLM(model=llm_model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        llm_kwargs['tokenizer'] = llm_kwargs['model'].get_tokenizer()

    # For each patient...
    results: List[Dict[str, Any]] = []
    stats:        List[UsageStat] = []
    patient_2_criteria_2_docs: Dict[str, Dict[str, List]] = {}
    counter_pa = 0
    
    for patient_idx, patient in enumerate(tqdm(dataset["patients"], desc='Looping through patients...')):
        counter_pa+=1

        # Query LLM for all patients
        results_i, stats_i = pipeline_koopman(dataset,
                                          patient,
                                          llm_model, 
                                          llm_kwargs, 
                                          is_exclude_rationale=is_exclude_rationale,
                                          is_add_global_decision= True)
        
        results.extend(results_i)
        stats.extend(stats_i)
        
    
    try:
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{file_name}.csv")
        print(f"Result Saved at {file_name}")
    except: 
        import pdb;pdb.set_trace()

    # Calculate overall token usage
    completion_tokens: int = sum([ stat.completion_tokens for stat in stats ])
    prompt_tokens:     int = sum([ stat.prompt_tokens for stat in stats ])
    
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
    elif llm_model in ['gpt-3.5-turbo-1106', 'shc-gpt-35-turbo-16k','gpt-3.5-turbo-0125']:
        cost_per_1k_completion_tokens = 0.0015
        cost_per_1k_prompt_tokens = 0.0020

    with open(f"{file_name}_usage.txt", 'w') as f:
        f.write(f"# of patients: {len(dataset)}\n")
        f.write(f"# of completion tokens: {completion_tokens}\n")
        f.write(f"# of prompt tokens: {prompt_tokens}\n")
        f.write(f"Total # of tokens: {completion_tokens + prompt_tokens}\n")
        f.write(f"Cost: ${(completion_tokens * cost_per_1k_completion_tokens / 1000 + prompt_tokens * cost_per_1k_prompt_tokens / 1000 )}\n")
    