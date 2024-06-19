import collections
import os
from dotenv import load_dotenv

from utils.pipelines import generate_result
load_dotenv()
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Dict, List, Any
from utils.data_loader import XMLDataLoader
from utils.helpers import CriterionAssessment, UsageStat

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a class-prevalence predictor')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    file_name = './outputs/baseline/baseline'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    path_to_train_data: str = './data/train'
    path_to_test_data: str = './data/n2c2-t1_gold_standard_test_data/test'
    
    # Load train data
    train_dataloader = XMLDataLoader(path_to_train_data)
    train_dataset = train_dataloader.load_data()
    
    # Determine prevalence of MET/NOT MET for each criterion, then choose most prevalent as prediction
    criterion_2_counts: Dict[str, Dict] = {
        criterion: collections.defaultdict(int)
        for criterion in train_dataloader.criteria
    }
    for patient in train_dataset:
        for criterion, is_met in patient['labels'].items():
            criterion_2_counts[criterion][is_met] += 1
    criterion_2_pred: Dict[str, int] = {
        criterion: 1 if criterion_2_counts[criterion][1] > criterion_2_counts[criterion][0] else 0
        for criterion in train_dataloader.criteria
    }
    print("Predictions: ", criterion_2_pred)
    with open(f"{file_name}_preds.txt", 'w') as f:
        f.write(str(criterion_2_pred))
    
    # Load test data
    test_dataloader = XMLDataLoader(path_to_test_data)
    test_dataset = test_dataloader.load_data()
    
    # For each patient...
    results: List[Dict[str, Any]] = []
    for patient_idx, patient in enumerate(tqdm(test_dataset, desc='Looping through patients...')):
        for criterion, pred in criterion_2_pred.items():
            assessment = CriterionAssessment(
                criterion=criterion,
                is_met=pred,
                confidence='high',
                rationale='',
                medications_and_supplements=[],
            )
            stat = UsageStat(completion_tokens=0, prompt_tokens=0)
            results.append(generate_result(patient, '', '', assessment, stat))
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{file_name}.csv")