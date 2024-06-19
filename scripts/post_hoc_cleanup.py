from typing import Dict, Any
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from utils.helpers import batch_query_hf, batch_query_openai
try:
    import vllm
except ImportError as e:
    pass  # ignore

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV containing outputs of eval.py')
    parser.add_argument('--is_exclude_rationale', action="store_true", help='If TRUE, then we ran the model without rationales, so use `completion_tokens` column for detecting NANs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    path_to_csv: str = args.path_to_csv
    is_exclude_rationale: bool = args.is_exclude_rationale
    file_name: str = path_to_csv.split('/')[-1].replace('.csv', '')
    llm_model: str = file_name.split('|')[0]
    tensor_parallel_size: int = 4 # for local HF models
    
    # Load results
    df = pd.read_csv(path_to_csv)

    # Determine strategy used
    strategy: str = ''
    is_all_criteria: bool = False
    is_all_notes: bool = False
    if '|all_criteria_all_notes|' in file_name:
        strategy = 'all_criteria_all_notes'
        is_all_criteria = True
        is_all_notes = True
    elif '|all_criteria_each_notes|' in file_name:
        strategy = 'all_criteria_each_notes'
        is_all_criteria = True
        is_all_notes = False
    elif '|each_criteria_all_notes|' in file_name:
        strategy = 'each_criteria_all_notes'
        is_all_criteria = False
        is_all_notes = True
    elif '|each_criteria_each_notes|' in file_name:
        strategy = 'each_criteria_each_notes'
        is_all_criteria = False
        is_all_notes = False
    else:
        raise ValueError('Could not determine strategy used')

    nan_col: str = 'rationale' if not is_exclude_rationale else 'completion_tokens'
    df_na = df[df[nan_col].isna()]
    print("start | # of NA rows:", len(df_na))
    
    # Query model to fill in NA rows
    if llm_model in ['gpt-4-1106-preview', 'gpt-4-32k', 'GPT4-32k', 'GPT4-32k-2', 'gpt-4', 'gpt-3.5-turbo-1106', 'shc-gpt-35-turbo-16k']:
        responses = batch_query_openai(df_na['prompt'].tolist(), llm_model, 'CriterionAssessments' if is_all_criteria else 'CriterionAssessment', n_procs=3, is_frequency_penalty=True)
    else:
        llm_kwargs: Dict[str, Any] = {}
        llm_kwargs['model'] = vllm.LLM(model=llm_model.replace('_', '/'), tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        llm_kwargs['tokenizer'] = llm_kwargs['model'].get_tokenizer()
        responses = batch_query_hf(df_na['prompt'].tolist(), llm_model, 'CriterionAssessments' if is_all_criteria else 'CriterionAssessment', llm_kwargs)

    import pickle
    pickle.dump(responses, open(f"{file_name}_responses.pkl", "wb"))
    # Inject new responses into df
    for idx in range(len(responses)):
        if responses[idx][0] is None: 
            continue
        row_idx = df_na.iloc[idx]['Unnamed: 0']
        assessments = responses[idx][0].assessments if hasattr(responses[idx][0], 'assessments') else [responses[idx][0]]
        for assessment in assessments:
            if assessment.criterion.lower().startswith(df.loc[row_idx, 'criterion'].lower()):
                df.loc[row_idx, 'rationale'] = assessment.rationale
                df.loc[row_idx, 'medications_and_supplements'] = str(assessment.medications_and_supplements)
                df.loc[row_idx, 'is_met'] = 1 if assessment.is_met else 0
                df.loc[row_idx, 'confidence'] = assessment.confidence
                df.loc[row_idx, 'completion_tokens'] = responses[idx][1].completion_tokens
                df.loc[row_idx, 'prompt_tokens'] = responses[idx][1].prompt_tokens
    df.to_csv(path_to_csv, index=False)
    
    print("start | # of NA rows:", len(df_na))
    print("end | # of NA rows:", len(df[df[nan_col].isna()]))
    
    if len(df[df[nan_col].isna()]) == 0:
        # Calculate overall token usage
        if is_all_criteria:
            prompt_tokens: int = df.drop_duplicates(['patient_id', 'prompt'])['prompt_tokens'].sum()
            completion_tokens: int = df.drop_duplicates(['patient_id', 'prompt'])['completion_tokens'].sum()
            api_calls: int = df.drop_duplicates(['patient_id', 'prompt']).shape[0]
        else:
            prompt_tokens: int = df['prompt_tokens'].sum()
            completion_tokens: int = df['completion_tokens'].sum()
            api_calls: int = df.shape[0]
            
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

        with open(os.path.join(os.path.dirname(path_to_csv), f"{file_name}_usage.txt"), 'w') as f:
            f.write(f"# of patients: {len(df['patient_id'].unique())}\n")
            f.write(f"# of completion tokens: {completion_tokens}\n")
            f.write(f"# of prompt tokens: {prompt_tokens}\n")
            f.write(f"# of API calls: {api_calls}\n")
            f.write(f"Total # of tokens: {completion_tokens + prompt_tokens}\n")
            f.write(f"Cost: ${(completion_tokens * cost_per_1k_completion_tokens / 1000 + prompt_tokens * cost_per_1k_prompt_tokens / 1000 )}\n")