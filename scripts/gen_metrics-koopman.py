import os
import re
import pandas as pd
from typing import Dict, Optional, List
import json
from utils.metrics import calc_metrics, plot_confusion_matrices
import argparse
from tqdm import tqdm
import numpy as np
import argparse

def gen_metrics(file_name: str):
    # Load results.csv
    file_name = file_name.replace(".csv", "")
    df = pd.read_csv(f"{file_name}.csv")

    # Logging
    if os.path.exists(f"{file_name}.log"):
        os.remove(f"{file_name}.log")


    force_boolean(df,'is_met')
    force_boolean(df,'true_label')
    grouped_df       = df.groupby(['patient_id', 'trail_id'])

    preds_from_single_criteria = []
    preds_from_global          = []
    true_labels                = []

    for patient_id  in df["patient_id"].unique():

        sub_df = df[df["patient_id"] == patient_id]

        for trail_id in sub_df["trail_id"].unique():
            sub_trail_df = sub_df[sub_df["trail_id"] == trail_id]
            inclussion_df:pd.DataFrame  = get_criterion_type(sub_df,'inclusion_criteria')
            exclussion_df:pd.DataFrame  = get_criterion_type(sub_df,'exclusion_criteria')
            meets:int                   = judge(inclussion_df,exclussion_df)

            preds_from_single_criteria.append(meets)
            preds_from_global.append(int(sub_trail_df["global_response"].values[0]))
            true_labels.append(int(sub_trail_df["true_label"].values[0]))

    metrics_local  = calc_metrics(np.array(true_labels), np.array(preds_from_single_criteria))
    metrics_global = calc_metrics(np.array(true_labels), np.array(preds_from_global))
    save_metrics_as_file(metrics_local,filename=f"{file_name}_local.txt")
    save_metrics_as_file(metrics_local,filename=f"{file_name}_global.txt")

def judge(inclussion_df,exclussion_df):
    all_inclussion:int = int(np.prod(inclussion_df["is_met"]))
    all_exclusion:int  = int(np.prod(exclussion_df["is_met"]))
    if all_inclussion == 1 and  all_exclusion == 0 :
        return 2
    else:
        return 0 
    
def force_boolean(df:pd.DataFrame,column:str) -> None:
     df[column] = df[column].apply(lambda x: 1 if x in [True, 'True',  1, '1'] else 0)
        
def get_criterion_type(df:pd.DataFrame,criteria=str) -> pd.DataFrame:
    return df[df['criterion'].str.contains(criteria, case=False)]

def save_metrics_as_file(metrics_dict, filename):
    with open(filename+".log", 'w') as file:
        for key, value in metrics_dict.items():
            file.write(json.dumps({key: value}) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate metrics from data.")
    parser.add_argument("--file_name", type=str, help="Name of the file")
    args = parser.parse_args()
    gen_metrics(args.file_name)

if __name__ == "__main__":
    main()



### python scripts/gen_metrics-koopman.py  --file_name outputs/2024-03-15-23-10-24/koopman_gpt-3.5-turbo-0125