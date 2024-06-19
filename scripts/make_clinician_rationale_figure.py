import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.data_loader import XMLDataLoader
sns.set_theme(style="whitegrid")
sns.set_context("paper")

BASE_DIR: str = "../"
PATH_TO_OUTPUT_DIR: str = os.path.join(BASE_DIR, "figures/")

# Load data
path_to_results_xlsx: str = os.path.join(BASE_DIR, "outputs/rationale_results/CT Patient Matching - Rationales Dataset.xlsx")
dfs = pd.read_excel(path_to_results_xlsx, sheet_name=["Dev", "Jenelle"])
df = pd.concat(dfs, axis=0)
df.reset_index(drop=True, inplace=True)

# Load ground truth data
df_orig = pd.read_csv(os.path.join(BASE_DIR, "outputs/rationale_dataset_gpt4--each_criteria_all_notes/stratified_sample.csv"))

# Prepare the data
df = df.merge(df_orig, left_on=['Patient ID', 'Criterion'], right_on=['patient_id', 'criterion'], how='inner', suffixes=('', '_orig'))
df['is_correct'] = (df['is_met'] == df['true_label']).astype(int)
grouped = df.groupby(['Criterion', 'Assessment', 'is_correct']).size()
grouped.to_csv('./clinical_rationale.csv')

# Function to plot the bar chart
color_mapping = {
    'Correct': 'tab:green',
    'Partially Correct': 'tab:olive',
    'Incorrect': 'tab:red',
    np.nan : 'grey',
}
def plot_clustered_bar(ax, data, label):
    # Create sub-dataframe for the given label
    sub_data = data.xs(label, level='is_correct')

    # Unique criteria and assessments
    criteria = sorted(df['Criterion'].unique(), reverse=True) # Sort A -> Z
    assessments = ['Incorrect', 'Partially Correct', 'Correct',]
    
    print(criteria)
    print(assessments)

    # Number of bars for each Criterion
    n_bars = len(assessments)

    # Positions of bars on y-axis
    barHeight = 0.2
    r = np.arange(len(criteria))
    positions = [r + barHeight*i for i in range(n_bars)]

    # Create bars
    for pos, assessment in zip(positions, assessments):
        counts = [sub_data.get((criterion, assessment), 0) for criterion in criteria]
        ax.barh(pos, counts, height=barHeight, edgecolor='grey', label=assessment, color=color_mapping[assessment])

    # General layout
    ax.set_yticks([r + barHeight*(n_bars-1)/2 for r in range(len(criteria))])
    ax.set_yticklabels(criteria, fontsize=12)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_xlim(0, 31)
    ax.tick_params(axis='x', labelsize=12)
    # ax.set_title(f'{"Model was Correct" if label else "Model was Incorrect"}', fontsize=12)
    if not label:
        # Only show legend for incorrect model b/c have space
        legend = ax.legend(title='Clinician Assessment', fontsize=12, title_fontsize=12, loc='upper right')
        legend.get_title().set_weight('bold')
    ax.yaxis.grid(False)

# Create a figure with two subplots
fig, ax = plt.subplots(figsize=(12, 6))
plot_clustered_bar(ax, grouped, 0)
plt.tight_layout()
plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, 'clinician_rationale_incorrect.pdf'))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
plot_clustered_bar(ax, grouped, 1)
plt.tight_layout()
plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, 'clinician_rationale_correct.pdf'))