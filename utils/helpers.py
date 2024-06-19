import pickle
from dotenv import load_dotenv
load_dotenv()
import json
import sys
import time
import os
import traceback
import chromadb
import httpx
from matplotlib import pyplot as plt
import concurrent.futures
import openai
import pandas as pd
from tqdm import tqdm
from utils.data_loader import XMLDataLoader
import torch 
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Tuple, Union, Optional
import sklearn.metrics
import numpy as np
from utils.outlines_helpers import JSONLogitsProcessor
from utils.types import CriterionAssessment, CriterionAssessments, CriterionAssessmentsForHF, UsageStat
import dirtyjson
from multiprocessing import Pool
import seaborn as sns

try:
    from vllm import SamplingParams
    PATH_TO_CHROMADB: str = '/share/pi/nigam/mwornow/pmct/data/chroma'
except ImportError as e:
    # PATH_TO_CHROMADB: str = '/Users/mwornow/Dropbox/Stanford/Shah Lab/Papers/pmct/data/chroma'
    PATH_TO_CHROMADB: str = '/Users/michaelwornow/Desktop/pmct/data/chroma'
    pass # ignore
import tiktoken

SYSTEM_PROMPT = "You are an expert clinical research coordinator tasked with assessing whether a patient meets the eligibility criteria of a clinical trial based on their clinical notes."

model_2_tokens = {
    'GPT4-32k' : 32_000,
    'GPT4-32k-2' : 32_000,
    'shc-gpt-35-turbo-16k' : 16_000,
    'Qwen/Qwen-72B-Chat' : 32_000,
    'mistralai/Mixtral-8x7B-v0.1' : 32_000,
    'togethercomputer/Llama-2-7B-32K-Instruct' : 32_000,
    'NousResearch/Yarn-Llama-2-70b-32k' : 32_000,
    'gpt-3.5-turbo-0125' : 4_000,
}

def compare_criterion_lists(list1, list2):
    # Convert lists to sets and check for equality
    set1 = set(list1)
    set2 = set(list2)

    if set1 == set2:
        return True
    else:
        return False

def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """Load HF model and tokenizer"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if model_name in ['mistralai/Mixtral-8x7B-v0.1', ]:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, use_flash_attention_2=True)
    else:
        model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

def load_collection(collection_name: str) -> chromadb.Collection:
    """Load Chroma db collection from disk"""
    path_to_chromadb: str = os.environ.get("PATH_TO_CHROMADB", PATH_TO_CHROMADB)
    client = chromadb.PersistentClient(path=path_to_chromadb)
    collection = client.get_collection(collection_name)
    return collection

def embed(text: str, model: AutoModel, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Embeds text using the given model and tokenizer"""
    device: str = model.device
    tokens: torch.Tensor = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    embedding: torch.Tensor = model(**tokens)[0][0].mean(dim=0).detach().cpu()
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding

def calc_metrics(true_labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    accuracy = sklearn.metrics.accuracy_score(true_labels, preds)
    precision = sklearn.metrics.precision_score(true_labels, preds)
    recall = sklearn.metrics.recall_score(true_labels, preds)
    micro_f1 = sklearn.metrics.f1_score(true_labels, preds, average='micro')
    met_f1 = sklearn.metrics.f1_score(true_labels, preds, average='binary')
    notmet_f1 = sklearn.metrics.f1_score(1-true_labels, 1-preds, average='binary')
    n2c2_f1 = (met_f1 + notmet_f1) / 2
    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'micro_f1' : micro_f1,
        'n2c2-met-f1' : met_f1,
        'n2c2-notmet-f1' : notmet_f1,
        # NOTE: This is the metric reported in the N2C2 challenge
        # It is incorrect -- it is actually calculating the Macro F1 score
        # for the two binary classes. The Micro F1 score calculated above
        # is the correct "Micro-F1" metric to use.
        # `n2c2_f1 = sklearn.metrics.f1_score(df['true_label'], df['is_met'], average='macro')`
        'n2c2-micro-f1' : n2c2_f1,
    }
    
def get_relevant_docs_for_criteria(dataloader: XMLDataLoader,
                                   collection: chromadb.Collection, 
                                    patient_id: int,
                                    model: AutoModel,
                                    tokenizer: AutoTokenizer,
                                    n_chunks: Optional[int],
                                    threshold: Optional[float],
                                    is_all_criteria: bool) -> Dict[str, List[Dict[str, Any]]]:
    """
        Given a patient, loop through all criteria and query Chroma db for relevant documents for each criterion.
        Return a dictionary of lists of documents that meet the similarity `threshold` for each criterion.
        
        If `is_all_criteria` is True, then return a single list of documents that match the combined embedding ('all') for all criteria.
    """
    criterion_2_results = {}
    
    # If using all criteria, then combine all criteria's definitions into one; otherwise, eval each criterion separately
    if is_all_criteria:
        all_criteria_definition: str = "\n".join([ f"- {criterion}: {definition}" for criterion, definition in dataloader.original_definitions.items() ])
        criteria: Dict[str, str] = { 'all' : all_criteria_definition }
    else:
        criteria: Dict[str, str] = dataloader.definitions

    for criterion, definition in criteria.items():
        criterion_2_results[criterion] = get_relevant_docs_for_criterion(definition,
                                                                         collection, 
                                                                         patient_id,
                                                                         model,
                                                                         tokenizer,
                                                                         n_chunks,
                                                                         threshold)
    return criterion_2_results

def get_relevant_docs_for_criterion(criterion_definition: str,
                                    collection: chromadb.Collection, 
                                    patient_id: int,
                                    model: AutoModel,
                                    tokenizer: AutoTokenizer,
                                    n_chunks: Optional[int],
                                    threshold: Optional[float]) -> List[Dict[str, Any]]:
    """
        Given a patient and criterion, query Chroma db for relevant documents for that criterion.
        Return the top-n_chunks documents.
    """
    assert n_chunks is None or n_chunks > 0, "n_chunks must be None or > 0"
    assert threshold is None or (threshold >= 0 and threshold <= 1), "threshold must be None or between 0 and 1"
    assert n_chunks is None or threshold is None, "n_chunks and threshold cannot both be specified"

    # Embed criteria's definition
    embedding = embed(criterion_definition, model, tokenizer).tolist()
    
    # Query Chroma
    if n_chunks == 9999:
        # Get all chunks
        results: Dict = collection.get(
            where={"patient_id": patient_id},
            include=["metadatas", "documents",  ],
        )
        results['ids'] = [ results['ids'] ]
        results['documents'] = [ results['documents'] ]
        results['metadatas'] = [ results['metadatas'] ]
        results['distances'] = [ [99] * len(results['ids'][0]) ]
    elif threshold is not None:
        results: List = collection.query(
            query_embeddings=embedding,
            where={"patient_id": patient_id},
            include=["metadatas", "documents", "distances", ],
        )
    else:
        results: Dict = collection.query(
            query_embeddings=embedding,
            where={"patient_id": patient_id},
            include=["metadatas", "documents", "distances", ],
            n_results=n_chunks,
        )

    # Filter results to only those >= similarity threshold
    records: List[Dict[str, Any]] = []
    for id, distance, text, metadata in zip(results['ids'][0], results['distances'][0], results['documents'][0], results['metadatas'][0]):
        similarity: float = 1 - distance
        if threshold is not None:
            if similarity < threshold:
                continue
        if len(text) < 40:
            # Ignore chunks < 40 chars b/c meaningless
            continue
        records.append({
            'id' : id,
            'metadata' : metadata,
            'similarity' : similarity,
            'text' : text,
        })

    return sorted(records, key=lambda x: (int(x['metadata']['note_idx']), int(x['metadata']['chunk_idx']))) # sort by note_idx, then chunk_idx

def prompt_to_llama(message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'

def prompt_to_qwen(message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return f'<|im_start|>{message}<|im_end|>\n'

def _batch_query_openai_worker(args):
    idx, prompt, llm_model, output_type, is_frequency_penalty, is_add_global_decision = args

    # OpenAI model
    llm_kwargs = {}
    if llm_model in ['GPT4-32k', 'GPT4-32k-2', 'shc-gpt-35-turbo-16k']:
        # Azure SHC
        llm_kwargs['openai_client'] = openai.AzureOpenAI(
            base_url=f"https://shcopenaisandbox.openai.azure.com/openai/deployments/{llm_model}/chat/completions?api-version=2023-07-01-preview",
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_version="2023-07-01-preview",
        )
        llm_kwargs['is_use_json'] = False
    else:
        llm_kwargs['openai_client'] = openai.OpenAI()
        llm_kwargs['is_use_json'] = True
    
    # Query LLM
    try:
        q = query_openai(prompt, 
                         llm_model, 
                         output_type, 
                         llm_kwargs, 
                         idx, 
                         n_retries=2, 
                         is_frequency_penalty=is_frequency_penalty,
                         is_add_global_decision =is_add_global_decision)
        os.makedirs('./outputs/', exist_ok=True)
        with open(f'./outputs/{idx}.pkl') as fd:
            pickle.dump(q, fd)
        return q
    except openai.RateLimitError:
        print("Rate limit exceeded -- waiting 30 seconds before retrying")
        time.sleep(30)
        return _batch_query_openai_worker(args)
    except httpx.HTTPStatusError:
        print("Rate limit exceeded -- waiting 30 seconds before retrying")
        time.sleep(30)
        return _batch_query_openai_worker(args)
    except openai.APIConnectionError:
        print("API connection error -- waiting 30 seconds before retrying")
        time.sleep(30)
        return _batch_query_openai_worker(args)
    except httpx.RemoteProtocolError:
        print("HTTP Protocol error -- waiting 30 seconds before retrying")
        time.sleep(30)
        return _batch_query_openai_worker(args)
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
        if is_add_global_decision:
            return None, None, None
        return None, None

def batch_query_openai(prompts: List[str], llm_model: str, output_type: str, n_procs: int = 5, is_frequency_penalty: bool = False, is_add_global_decision: bool = False) -> List[Tuple[Union[CriterionAssessment, CriterionAssessments], UsageStat]]:
    tasks = [ (idx, prompt, llm_model, output_type, is_frequency_penalty, is_add_global_decision) for idx, prompt in enumerate(prompts) ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_procs) as pool:
        results: List[Tuple[Union[CriterionAssessment, CriterionAssessments], UsageStat]] = list(tqdm(pool.map(_batch_query_openai_worker, tasks), desc='Running batch_query_openai()...', total=len(tasks)))
    return results

def batch_query_hf(prompts: Union[List[str], str], llm_model: str, output_type: str, llm_kwargs: Dict[str, Any], n_retries: int = 0) -> List[Tuple[Union[CriterionAssessment, CriterionAssessments], UsageStat]]:
    """Queries local HF model with Outlines and returns the JSON response."""
    model = llm_kwargs['model']
    tokenizer = llm_kwargs['tokenizer']
    
    # Load prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    if 'llama' in llm_model.lower() and n_retries == 0:
        prompts = [ prompt_to_llama(x) for x in prompts ]
    elif 'qwen' in llm_model.lower() and n_retries == 0:
        prompts = [ prompt_to_qwen(x) for x in prompts ]
    
    # Generate outputs
    schema = CriterionAssessment.model_json_schema() if output_type == 'CriterionAssessment' else CriterionAssessmentsForHF.model_json_schema()
    sampling_params = SamplingParams(max_tokens=10_000, 
                                    logits_processors=[JSONLogitsProcessor(schema, model)], 
                                    use_beam_search=False,
                                    n=1,
                                    temperature=0 if n_retries < 1 else 0.1,
                                    frequency_penalty=0 if n_retries < 1 else 0.1,
    )
    raw_responses: List[Dict[str, Any]] = model.generate(prompts, sampling_params)
    
    # Parse
    requeue: List[str] = []
    results: List[Tuple[Union[CriterionAssessment, CriterionAssessmentsForHF], UsageStat]] = []
    for raw_response, prompt in zip(raw_responses, prompts):
        if raw_response is None:
            results.append((None, None))
        else:
            output = raw_response.outputs[0].text
            try:
                # # For Qwen
                # output = output.replace('<|im_start|>', '').replace('<|im_end|>', '')
                # output = output[:output.rfind('}')].strip()
                response: Dict[str, str] = dirtyjson.loads(output)
                result = CriterionAssessment(**response) if output_type == 'CriterionAssessment' else CriterionAssessmentsForHF(**response)
                # Get usage stats
                stat: UsageStat = UsageStat(completion_tokens=len(tokenizer.encode(output)),
                                            prompt_tokens=len(tokenizer.encode(prompt)))
                results.append((result, stat))
            except Exception as e:
                print(str(e))
                with open('/share/pi/nigam/mwornow/pmct/outputs/redos.txt', 'a') as fd:
                    fd.write(str(output) + '\n\n')
                requeue.append(prompt)
    
    new_results: List[Tuple[Union[CriterionAssessment, CriterionAssessmentsForHF], UsageStat]] = [ (None, None) for x in requeue ]
    if len(requeue) > 0 and n_retries < 2:
        print(f"Requeuing {len(requeue)} prompts")
        new_results = batch_query_hf(requeue, llm_model, output_type, llm_kwargs, n_retries + 1)
    all_results = results + new_results
    
    # Convert CriterionAssessmentsForHF to CriterionAssessments
    if output_type == 'CriterionAssessments':
        all_results = [ (CriterionAssessments(assessments=[
            CriterionAssessment(criterion=x[0].abdominal.criterion, medications_and_supplements=[], rationale=x[0].abdominal.rationale, is_met=x[0].abdominal.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].advanced_cad.criterion, medications_and_supplements=[], rationale=x[0].advanced_cad.rationale, is_met=x[0].advanced_cad.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].alcohol_abuse.criterion, medications_and_supplements=[], rationale=x[0].alcohol_abuse.rationale, is_met=x[0].alcohol_abuse.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].asp_for_mi.criterion, medications_and_supplements=[], rationale=x[0].asp_for_mi.rationale, is_met=x[0].asp_for_mi.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].creatinine.criterion, medications_and_supplements=[], rationale=x[0].creatinine.rationale, is_met=x[0].creatinine.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].dietsupp_2mos.criterion, medications_and_supplements=[], rationale=x[0].dietsupp_2mos.rationale, is_met=x[0].dietsupp_2mos.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].drug_abuse.criterion, medications_and_supplements=[], rationale=x[0].drug_abuse.rationale, is_met=x[0].drug_abuse.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].english.criterion, medications_and_supplements=[], rationale=x[0].english.rationale, is_met=x[0].english.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].keto_1yr.criterion, medications_and_supplements=[], rationale=x[0].keto_1yr.rationale, is_met=x[0].keto_1yr.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].hba1c.criterion, medications_and_supplements=[], rationale=x[0].hba1c.rationale, is_met=x[0].hba1c.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].major_diabetes.criterion, medications_and_supplements=[], rationale=x[0].major_diabetes.rationale, is_met=x[0].major_diabetes.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].makes_decisions.criterion, medications_and_supplements=[], rationale=x[0].makes_decisions.rationale, is_met=x[0].makes_decisions.is_met, confidence='high'),
            CriterionAssessment(criterion=x[0].mi_6mos.criterion, medications_and_supplements=[], rationale=x[0].mi_6mos.rationale, is_met=x[0].mi_6mos.is_met, confidence='high'),
        ]), x[1]) if x[0] is not None and isinstance(x[0], CriterionAssessmentsForHF) else (x if x[0] is not None else (None, None)) for x in all_results ]
    
    return all_results

def query_openai(prompt:      str, 
                 llm_model:   str, 
                 output_type: str, 
                 llm_kwargs: Dict[str, Any], 
                 idx:        int, 
                 n_retries: int = 4, 
                 is_frequency_penalty: bool = False,
                 is_add_global_decision: bool = False ) -> Tuple[Union[CriterionAssessment, CriterionAssessments], UsageStat]:
    """Queries OpenAI model with the given prompt and returns the JSON response."""
    client            = llm_kwargs['openai_client']
    is_use_json: bool = llm_kwargs['is_use_json']

    # Calculate max tokens
    encoding              = tiktoken.get_encoding('cl100k_base')
    model_max_tokens: int = model_2_tokens[llm_model]
    max_tokens: int       = len(encoding.encode(prompt))
    max_tokens: int = 4000
    

    if is_use_json:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=10_000 if llm_model in [ 'GPT4-32k', 'GPT4-32k-2' ] else min(max(1, model_max_tokens-max_tokens), 3000),
            model=llm_model,
            temperature=0 if n_retries < 1 else 0.1,
        )
        result: Dict[str, str] = json.loads(response.choices[0].message.content)
    else:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10_000 if llm_model in [ 'GPT4-32k', 'GPT4-32k-2' ] else min(max(1, model_max_tokens-max_tokens), 4096), # if not is_frequency_penalty else 10_000),
                model=llm_model,
                temperature=0 if n_retries < 1 else 0.1,
                frequency_penalty=0 if not is_frequency_penalty else (0 if n_retries < 1 else 0.1),
            )
            result = response.choices[0].message.content.replace("```json", "").replace("\n```", "").replace("```", "").strip("\n")
            result = result[result.index("{"):]
            result: Dict[str, str] = dirtyjson.loads(result)
            
        except openai.RateLimitError as e:
            raise e
        except openai.APIError as e:
            raise e
        except Exception as e:
            print(f"Error decoding JSON response. Trying again. idx={idx} | n_retries={n_retries}...")
            with open("./result.txt", "a") as fd:
                fd.write(str(result) + '\n\n')
            if n_retries < 1:
                return query_openai(prompt, llm_model, output_type, llm_kwargs, idx, n_retries = n_retries + 1, is_frequency_penalty=is_frequency_penalty)
            elif n_retries < (2 if not is_frequency_penalty else 4):
                print(str(e))
                return query_openai(prompt, llm_model, output_type, llm_kwargs, idx, n_retries = n_retries + 1, is_frequency_penalty=is_frequency_penalty)
            else:
                print('Giving up...')
                if is_add_global_decision:
                    return None, None, None
                return None, None
    
    # Get usage stats
    stats: UsageStat = UsageStat(completion_tokens=response.usage.completion_tokens, 
                                    prompt_tokens=response.usage.prompt_tokens)
    if output_type == 'CriterionAssessment':
        if str(type(result)) == "<class 'dirtyjson.attributed_containers.AttributedDict'>":
            result = { key: val for key, val in result.items() }
        return CriterionAssessment(**result), stats
    
    else:
        try:
            global_result = int(result['global_decision'])
        except:
            global_result = 3
        if str(type(result)) == "<class 'dirtyjson.attributed_containers.AttributedList'>":
            result = {'assessments' : [{ key: val for key, val in x.items() } for x in result['assessments']]}
        if is_add_global_decision:
            return CriterionAssessments(**result), stats, global_result
        else:
            return CriterionAssessments(**result), stats

def plot_confusion_matrices(df: pd.DataFrame, file_name: Optional[str] = None):
    # Unique labels
    unique_labels = df['criterion'].unique()
    num_labels = len(unique_labels)

    # Creating subplots
    cols = 3
    rows = (num_labels + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    for i, label in enumerate(unique_labels):
        # Create binary classification problem
        y_binary = df[df['criterion'] == label]['true_label']
        y_hat_binary = df[df['criterion'] == label]['is_met']

        # Compute confusion matrix
        cm = sklearn.metrics.confusion_matrix(y_binary, y_hat_binary)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Confusion Matrix for Label: {label}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Turn off any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()