import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils.helpers import calc_metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_context("paper")

# Taken from: https://academic.oup.com/jamia/article/26/11/1247/5568257?login=true (Table 3)
SOTA_N2C2_MICRO_SCORE: float = 0.9100
SOTA_N2C2_MACRO_SCORE: float = 0.7525
SOTA_PRECISION: float = 0.88
SOTA_RECALL: float = 0.91

# Paths
BASE_DIR: str = "./outputs-results/"
PATH_TO_OUTPUT_DIR: str = "./figures/"

def to_human_friendly(n: float) -> str:
    return "{0:.2f}M".format(n / 1e6)

def get_file_with_end(directory: str, file_end: str) -> Optional[str]:
    # List all files in the directory
    files = os.listdir(directory)
    # Check if any file ends with "file_end"
    for file in files:
        if file.endswith(file_end):
            return os.path.join(directory, file)
    return None

def get_path_to_preds_csv(dir: str) -> str:
    if get_file_with_end(dir, "_preds.csv") is None:
        path_to_csv: str = get_file_with_end(dir, ".csv")
        _ = os.system(f"python scripts/gen_metrics.py '{path_to_csv}'")
    path_to_csv: str = get_file_with_end(dir, '_preds.csv')
    return path_to_csv

def gen_metrics_df(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Output format:
        | df (key in `dfs`) | criterion | metric | value |
        | gpt-3 | overall | acc | 0.5 |
        | gpt-4 | overall | f1 | 0.5 |
        | gpt-3 | ABDOMINAL | precision | 0.5 |
    """
    rows = []
    for df_key, df in dfs.items():
        # Get n2c2 macro
        met_f1s, notmet_f1s = [], []
        # Specific criteria
        all_bootstraps = {}
        for criterion in df['criterion'].unique():
            metrics, bootstrap_ranges, bootstraps = calc_metrics(df[df['criterion'] == criterion]['true_label'].values, df[df['criterion'] == criterion]['is_met'].values)
            for k, v in metrics.items():
                rows.append({
                    'df' : df_key,
                    'criterion' : criterion,
                    'metric' : k,
                    'value' : v,
                    'lower' : bootstrap_ranges[k]['lower'],
                    'upper' : bootstrap_ranges[k]['upper'],
                })
                if k == 'n2c2-met-f1': met_f1s.append(v)
                if k == 'n2c2-notmet-f1': notmet_f1s.append(v)
            all_bootstraps[criterion] = bootstraps
        macro_f1_bootstraps: List[float] = [
            (
                sum([ all_bootstraps[criterion]['n2c2-met-f1'][i] for criterion in all_bootstraps.keys() ])
                + sum([ all_bootstraps[criterion]['n2c2-notmet-f1'][i] for criterion in all_bootstraps.keys() ])
            ) / (2 * len(all_bootstraps.keys()))
            for i in range(1000)
        ]
        rows.append({
            'df' : df_key,
            'criterion' : 'overall',
            'metric' : 'n2c2-macro-f1',
            'value' : (sum(met_f1s) + sum(notmet_f1s)) / (len(met_f1s) + len(notmet_f1s)),
            'lower' : np.percentile(macro_f1_bootstraps, 2.5),
            'upper' : np.percentile(macro_f1_bootstraps, 97.5),
        })
        # Calculate overall metrics
        metrics, bootstrap_ranges, __ = calc_metrics(df['true_label'].values, df['is_met'].values)

        for k, v in metrics.items():
            rows.append({
                'df' : df_key,
                'criterion' : 'overall',
                'metric' : k,
                'value' : v,
                'lower' : bootstrap_ranges[k]['lower'],
                'upper' : bootstrap_ranges[k]['upper'],
            })
    return pd.DataFrame(rows)

def add_usage_stats_to_df(df: pd.DataFrame, path_to_dir: str) -> pd.DataFrame:
    # Add in usage stats
    stats = []
    for key in df['df'].unique():
        path_to_txt: str = get_file_with_end(os.path.join(path_to_dir, key), '_usage.txt') 
        with open(path_to_txt, 'r') as f:
            content = f.read().replace('\n', ' ')
            completion_tokens: int = re.search(r'# of completion tokens: (\d+)', content).group(1)
            prompt_tokens: int = re.search(r'# of prompt tokens: (\d+)', content).group(1)
            cost: int = re.search(r'Cost: \$(\d+\.?\d*)', content).group(1)
            # api_calls: int = re.search(r'# of API calls: (\d+)', content).group(1) if '# of API calls:' in content[5] else -1
            stats.append({
                'df' : key,
                'criterion' : 'overall',
                'metric' : 'completion_tokens',
                'value' : completion_tokens,
            })
            stats.append({
                'df' : key,
                'criterion' : 'overall',
                'metric' : 'prompt_tokens',
                'value' : prompt_tokens,
            })
            stats.append({
                'df' : key,
                'criterion' : 'overall',
                'metric' : 'cost',
                'value' : cost,
            })
            # stats.append({
            #     'df' : key,
            #     'criterion' : 'overall',
            #     'metric' : 'api_calls',
            #     'value' : api_calls,
            # })
            stats.append({
                'df' : key,
                'criterion' : 'overall',
                'metric' : 'tokens',
                'value' : int(prompt_tokens) + int(completion_tokens),
            })
    df = pd.concat([df, pd.DataFrame(stats)], axis=0)
    return df

def save_df_to_latex(df: pd.DataFrame, 
                     file_name: str, 
                     caption: str, 
                     index: str = 'df', 
                     other_cols_to_keep: List[str] = [],
                     model_2_name: Dict[str, str] = None,
                     is_include_95_ci: bool = True) -> pd.DataFrame:
    # Add CI interval
    if is_include_95_ci:
        df['value'] = df.apply(lambda row: f"${round(row['value'], 2)}_{{({round(row['lower'], 2)},{round(row['upper'], 2)})}}$" if not pd.isna(row['upper']) else row['value'], axis=1)
    # Pivot
    df = df.pivot(index=index, columns='metric', values='value').reset_index()
    # Rename models
    if model_2_name is not None:
        df['df'] = df['df'].map(model_2_name)
    # Reorder columns
    other_cols = [col for col in df.columns if col not in ['df', 'precision', 'recall', 'n2c2-micro-f1', 'n2c2-macro-f1']]
    df = df[['df'] + other_cols + [ 'precision', 'recall', 'n2c2-micro-f1', 'n2c2-macro-f1']]
    df.columns = ['Model',] + other_cols + ['Prec.', 'Rec.', 'n2c2-Micro-F1', 'n2c2-Macro-F1' ]
    # Keep certain columns
    df = df[['Model', 'Prec.', 'Rec.', 'n2c2-Macro-F1', 'n2c2-Micro-F1' ] + other_cols_to_keep]
    if 'completion_tokens' in df.columns:
        df['completion_tokens'] = df['completion_tokens'].astype(int)
    if 'prompt_tokens' in df.columns:
        df['prompt_tokens'] = df['prompt_tokens'].astype(int)
    if 'tokens' in df.columns:
        df['tokens'] = df['tokens'].astype(int).apply(lambda x: to_human_friendly(x))
    if 'cost' in df.columns:
        df['cost'] = df['cost'].astype(float).apply(lambda x: f"\${x:.2f}")
    # Save to LaTeX
    df.to_latex(os.path.join(PATH_TO_OUTPUT_DIR, f'{file_name}.tex'), 
                    index=False, 
                    float_format="%.2f",
                    label=f'tab:{file_name}',
                    caption=caption)

def plot_thresholds_line_plot(df: pd.DataFrame, df_strat: pd.DataFrame, x_metric: str, y_metric: str, title: str):
    # Make plot
    # x-axis = chunk
    # y-axis = n2c2-f1
    # hue = llm
    df_plot = pd.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'n2c2-micro-f1')], 
                    df[(df['criterion'] == 'overall') & (df['metric'] == 'cost')][['df', 'metric', 'value']], 
                    on=['df'], 
                    suffixes=('_micro_f1', '_cost'))
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'completion_tokens')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_completion_tokens', 'value' : 'value_completion_tokens'}), 
                            on=['df'])
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'prompt_tokens')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_prompt_tokens', 'value' : 'value_prompt_tokens'}), 
                            on=['df'])
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'n2c2-macro-f1')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_macro_f1', 'value' : 'value_macro_f1'}), 
                            on=['df'])
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'precision')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_precision', 'value' : 'value_precision'}), 
                            on=['df'])
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'recall')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_recall', 'value' : 'value_recall'}), 
                            on=['df'])
    df_plot = df_plot.merge(df[(df['criterion'] == 'overall') & (df['metric'] == 'tokens')][['df', 'metric', 'value']].rename(columns={'metric' : 'metric_tokens', 'value' : 'value_tokens'}), 
                            on=['df'])
    df_plot = df_plot.rename(columns={
        'value_micro_f1' : 'n2c2-micro-f1', 
        'value_macro_f1' : 'n2c2-macro-f1', 
        'value_precision' : 'precision', 
        'value_recall' : 'recall', 
        'value_cost' : 'cost', 
        'value_tokens' : 'tokens', 
        'value_completion_tokens' : 'completion_tokens', 
        'value_prompt_tokens' : 'prompt_tokens',
    })
    # reformat cols
    df_plot['cost'] = df_plot['cost'].astype(float)
    df_plot['chunk'] = df_plot['chunk'].str.replace("c=", "").astype(int)
    # make plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_plot, x=x_metric, y=y_metric, hue='llm', style='embed_model', markers=True, ax=ax)
    ax.set_ylabel(y_metric)
    ax.set_xlabel(x_metric)
    # ax.legend(title='Base LLM+Embed Model', loc='lower center')
    ax.get_legend().remove()
    # SOTA
    if y_metric == 'n2c2-micro-f1':
        plt.axhline(SOTA_N2C2_MICRO_SCORE, linestyle='--', color='green')
        plt.ylim(0.73, 0.95)
    elif y_metric == 'n2c2-macro-f1':
        plt.axhline(SOTA_N2C2_MACRO_SCORE, linestyle='--', color='green')
        plt.ylim(0.47, 0.87)
    # Best Model
    for llm in df['llm'].unique():
        df_strat_llm = df_strat[df_strat['df'].str.startswith(llm) & (df_strat['criterion'] == 'overall')]
        plt.plot(float(df_strat_llm[df_strat_llm['metric'] == x_metric]['value'].iloc[0]),
                float(df_strat_llm[df_strat_llm['metric'] == y_metric]['value'].iloc[0]),
                marker='*', markersize=8, color='tab:orange' if llm == 'gpt4' else 'tab:blue',
        )
    # Add 
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, f'{title}__{x_metric}_v_{y_metric}.png'))

if __name__ == '__main__':
    
    # Prompt Strategy
    llms = [ 'gpt3.5', 'gpt4' ]
    strats = ['all_criteria_each_notes', 'all_criteria_all_notes', 'each_criteria_all_notes', 'each_criteria_each_notes']
    path_to_dir: str = os.path.join(BASE_DIR, 'prompt-strat/')
    dfs_strat: Dict[str, pd.DataFrame] = {}
    for llm in llms:
        for strat in strats:
            dir: str = os.path.join(path_to_dir, f'{llm}-{strat}')
            path_to_csv: str = get_path_to_preds_csv(dir)
            df_ = pd.read_csv(path_to_csv)
            dfs_strat[f'{llm}-{strat}'] = df_
    assert len(dfs_strat) == 8
    df_strat = gen_metrics_df(dfs_strat)
    df_strat = add_usage_stats_to_df(df_strat, path_to_dir)
    df_strat['criteria'] = df_strat['df'].map(lambda x: 'All' if 'all_criteria' in x.split('-')[1] else 'Individual')
    df_strat['notes'] = df_strat['df'].map(lambda x: 'All' if 'all_notes' in x.split('-')[1] else 'Individual')
    save_df_to_latex(
        df_strat[df_strat['criterion'] == 'overall'],
        "prompt_strat",
        "Simultaneously considering multiple notes/criteria hurts performance.",
        index=['df', 'criteria', 'notes'],
        other_cols_to_keep=['cost', 'tokens'],
        model_2_name={
            f'{llm}-{strat}' : f"{'GPT-3.5' if llm == 'gpt3.5' else 'GPT-4'} {strat}"
            for llm in llms
            for strat in strats
        }
    )
    
    exit()
    
    # Chunks
    llms = [ 'gpt3.5', 'gpt4' ]
    strats = ['all_criteria_each_notes', 'all_criteria_all_notes', 'each_criteria_all_notes', 'each_criteria_each_notes']
    embed_models = ['bge', 'mpnet', 'minilm',]
    chunks = [1, 3, 5, 10, ] #9999]

    for strat in strats:
        path_to_dir: str = os.path.join(BASE_DIR, f'chunks/{strat}')
        if not os.path.exists(path_to_dir):
            continue
        df_chunks: Dict[str, pd.DataFrame] = {}
        for llm in llms:
            for embed_model in embed_models:
                for chunk in chunks:
                    dir: str = os.path.join(path_to_dir, f'{llm}-{embed_model}-c={chunk}')
                    if not os.path.exists(dir):
                        continue
                    path_to_csv: str = get_path_to_preds_csv(dir)
                    df_ = pd.read_csv(path_to_csv)
                    df_chunks[f'{llm}-{embed_model}-c={chunk}'] = df_
        df = gen_metrics_df(df_chunks)
        df = add_usage_stats_to_df(df, path_to_dir)
        # Split llm/embed_model/chunk
        df['llm'] = df['df'].map(lambda x: x.split('-')[0])
        df['embed_model'] = df['df'].map(lambda x: x.split('-')[1])
        df['chunk'] = df['df'].map(lambda x: x.split('-')[2].replace('t=', ''))
        save_df_to_latex(
            df[df['criterion'] == 'overall'],
            "chunks",
            "Efficiency gains from using a two-stage retrieval-based pipeline",
            index=['df', 'llm', 'embed_model', 'chunk'],
        )
        for x_metric in [ 'cost', 'tokens']: # 'chunk',
            for y_metric in ['n2c2-micro-f1', 'n2c2-macro-f1', ]:
                plot_thresholds_line_plot(df, 
                                          df_strat[
                                                (df_strat['criteria'] == ('All' if 'all_criteria' in strat else 'Individual')) &
                                                (df_strat['notes'] == ('All' if 'all_notes' in strat else 'Individual'))
                                            ], 
                                          x_metric, 
                                          y_metric, 
                                          f'{strat}__chunk_')

    # LLM Models
    df_llms: Dict[str, pd.DataFrame] = {
        'baseline' : pd.read_csv(os.path.join(BASE_DIR, 'baseline/baseline_preds.csv')),
        'gpt35' : pd.read_csv(os.path.join(BASE_DIR, 'prompt-strat/gpt3.5-all_criteria_each_notes/shc-gpt-35-turbo-16k|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
        'gpt4' : pd.read_csv(os.path.join(BASE_DIR, 'prompt-strat/gpt4-all_criteria_each_notes/GPT4-32k|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
        'mixtral' : pd.read_csv(os.path.join(BASE_DIR, 'llm_models/mixtral-32k/mistralai_Mixtral-8x7B-Instruct-v0.1|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
        'llama2' : pd.read_csv(os.path.join(BASE_DIR, 'llm_models/llama2-70b/NousResearch_Yarn-Llama-2-70b-32k|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
    }
    df = gen_metrics_df(df_llms)
    df = pd.concat([df, pd.DataFrame([
        { 
            'df' : 'sota', 
            'criterion' : 'overall', 
            'metric' : 'n2c2-micro-f1', 
            'value' : SOTA_N2C2_MICRO_SCORE 
        },
        { 
            'df' : 'sota', 
            'criterion' : 'overall', 
            'metric' : 'n2c2-macro-f1', 
            'value' : SOTA_N2C2_MACRO_SCORE 
        },
        { 
            'df' : 'sota', 
            'criterion' : 'overall', 
            'metric' : 'precision', 
            'value' : SOTA_PRECISION, 
        },
        { 
            'df' : 'sota', 
            'criterion' : 'overall', 
            'metric' : 'recall', 
        },
    ])], ignore_index=True)
    save_df_to_latex(
        df[df['criterion'] == 'overall'],
        "base_llms",
        "Zero-shot results on the 2018 n2c2 cohort selection challenge.",
        model_2_name={
            'gpt35' : 'GPT-3.5',
            'gpt4' : 'GPT-4',
            'llama2' : 'Llama-2-70b',
            'meditron' : 'Meditron-70b',
            'mixtral' : 'Mixtral-8x7b',
            'sota' : 'Prior SOTA',
            'baseline' : 'Class Prevalence',
        },
    )
    
    
    # 1-shot
    llms = [ 'gpt4' ]
    strats = ['all_criteria_each_notes', 'all_criteria_all_notes', 'each_criteria_all_notes', 'each_criteria_each_notes']
    path_to_dir: str = os.path.join(BASE_DIR, '1-shot/')
    dfs_strat: Dict[str, pd.DataFrame] = {}
    for llm in llms:
        for strat in strats:
            dir: str = os.path.join(path_to_dir, f'{llm}-{strat}')
            path_to_csv: str = get_path_to_preds_csv(dir)
            df_ = pd.read_csv(path_to_csv)
            dfs_strat[f'{llm}-{strat}'] = df_
    df_strat = gen_metrics_df(dfs_strat)
    df_strat = add_usage_stats_to_df(df_strat, path_to_dir)
    df_strat['criteria'] = df_strat['df'].map(lambda x: 'All' if 'all_criteria' in x.split('-')[1] else 'Individual')
    df_strat['notes'] = df_strat['df'].map(lambda x: 'All' if 'all_notes' in x.split('-')[1] else 'Individual')
    save_df_to_latex(
        df_strat[df_strat['criterion'] == 'overall'],
        "1_shot",
        "1-shot results",
        index=['df', 'criteria', 'notes'],
        other_cols_to_keep=['cost', 'tokens'],
        model_2_name={
            f'{llm}-{strat}' : f"{'GPT-3.5' if llm == 'gpt3.5' else 'GPT-4'} {strat}"
            for llm in llms
            for strat in strats
        }
    )
    
    # Rationale Ablation
    llms = [ 'gpt3.5', 'gpt4' ]
    strats = ['all_criteria_each_notes', 'all_criteria_all_notes', 'each_criteria_all_notes', 'each_criteria_each_notes']
    path_to_dir: str = os.path.join(BASE_DIR, 'ablation-rationale/')
    dfs_strat: Dict[str, pd.DataFrame] = {}
    for llm in llms:
        for strat in strats:
            dir: str = os.path.join(path_to_dir, f'{llm}-{strat}')
            if not os.path.exists(dir):
                continue
            path_to_csv: str = get_path_to_preds_csv(dir)
            df_ = pd.read_csv(path_to_csv)
            dfs_strat[f'{llm}-{strat}'] = df_
    df_strat = gen_metrics_df(dfs_strat)
    df_strat = add_usage_stats_to_df(df_strat, path_to_dir)
    df_strat['criteria'] = df_strat['df'].map(lambda x: 'All' if 'all_criteria' in x.split('-')[1] else 'Individual')
    df_strat['notes'] = df_strat['df'].map(lambda x: 'All' if 'all_notes' in x.split('-')[1] else 'Individual')
    save_df_to_latex(
        df_strat[df_strat['criterion'] == 'overall'],
        "ablation_rationale",
        "Without Rationale.",
        index=['df', 'criteria', 'notes'],
        other_cols_to_keep=['cost', 'tokens'],
        model_2_name={
            f'{llm}-{strat}' : f"{'GPT-3.5' if llm == 'gpt3.5' else 'GPT-4'} {strat}"
            for llm in llms
            for strat in strats
        }
    )


    # Prompt Engineering
    dfs_eng: Dict[str, pd.DataFrame] = {
        'gpt35-orig_def' : pd.read_csv(os.path.join(BASE_DIR, 'prompt-engineering/gpt3.5-all_criteria_each_notes--use_orig_defs/shc-gpt-35-turbo-16k|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
        'gpt4-orig_def' : pd.read_csv(os.path.join(BASE_DIR, 'prompt-engineering/gpt4-all_criteria_each_notes--use_orig_defs/GPT4-32k|sentence-transformers_all-MiniLM-L6-v2|9999|all_criteria_each_notes|test|criteria-all_preds.csv')),
    }
    df = gen_metrics_df(dfs_eng)
    save_df_to_latex(
        df[df['criterion'] == 'overall'],
        "prompt_eng",
        "Increasing the specificity of criterion definitions improves model accuracy.",
        model_2_name={
            'gpt35-orig_def' : 'GPT-3.5 Original',
            'gpt35-new_def' : 'GPT-3.5 Improved',
            'gpt4-orig_def' : 'GPT-4 Original',
            'gpt4-new_def' : 'GPT-4 Improved',
        },
    )
