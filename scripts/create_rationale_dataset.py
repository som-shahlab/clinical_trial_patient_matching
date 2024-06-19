import argparse
import os
from dotenv import load_dotenv

from utils.data_loader import XMLDataLoader
load_dotenv()
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate the model on the given data')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV containing outputs of eval.py')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    path_to_csv = args.path_to_csv
    base_dir: str = os.path.dirname(path_to_csv)
    path_to_output_dir: str = os.path.join(base_dir, 'rationale_dataset')
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load results
    dataloader = XMLDataLoader('./data/train')
    df = pd.read_csv(path_to_csv)
    df['is_correct'] = df['is_met'] == df['true_label']
    df['stratification_group'] = df['is_correct'].astype(str) + '|' + df['criterion'].astype(str)

    # Stratified sampling
    n: int = 30
    stratified_sample = df.groupby('stratification_group', group_keys=False).apply(lambda x: x.sample(min(len(x), n), random_state=1))
    stratified_sample.to_csv(os.path.join(path_to_output_dir, 'stratified_sample.csv'), index=False)

    for idx, row in stratified_sample.iterrows():
        note = row['note']
        criterion = row['criterion']
        rationale = row['rationale']
        true_label = row['true_label']
        is_met = row['is_met']
        patient_id = row['patient_id']
        os.makedirs(os.path.join(path_to_output_dir, str(patient_id)), exist_ok=True)
        with open(os.path.join(path_to_output_dir, str(patient_id), f'{idx}.txt'), 'w') as f:
            f.write(f'Patient ID: {patient_id} | Data Idx: {idx}\n')
            f.write('\n====================\n\n')
            f.write(f'Inclusion Criterion: {criterion}\n')
            f.write(f'Definition: {dataloader.original_definitions[criterion]}\n')
            f.write('\n====================\n\n')
            f.write(f'Rationale: {rationale}\n')
            f.write('\n====================\n\n')
            f.write(f'Notes:\n\n{note.strip()}\n')
    
    with open(os.path.join(path_to_output_dir, 'README.md'), 'w') as f:
        f.write(f"Command run: `python scripts/gen_rationale_dataset.py {path_to_csv}`\n")
        f.write(f"Wrote {len(stratified_sample)} files to {path_to_output_dir}\n")
        f.write(f"Number of unique patients: {stratified_sample['patient_id'].nunique()}\n")
        f.write("Count per criterion:\n")
        f.write(str(stratified_sample['criterion'].value_counts()))