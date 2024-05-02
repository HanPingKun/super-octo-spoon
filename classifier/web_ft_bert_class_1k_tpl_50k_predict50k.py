import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import logging
import random
import pkg_resources
import sklearn
from rxnfp.models import SmilesClassificationModel
logger = logging.getLogger(__name__)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
with open('rxnclass2id.json', 'r') as f:
    rxnclass2id = json.load(f)
with open('rxnclass2name.json', 'r') as f:
    rxnclass2name = json.load(f)
all_classes =sorted(rxnclass2id.keys())
train_model_path =  'bert_class_1k_tpl+50k'
model = SmilesClassificationModel("bert", train_model_path, use_cuda=torch.cuda.is_available())
def process_smiles(smiles):
    y_preds = model.predict(smiles)
    df_preds = pd.DataFrame(y_preds)
    predicted_class_id = df_preds.iloc[[0],[0]].values[0].item()
    def get_class_name_from_number(number):
            # 反转rxnclass2id字典，以便根据数字ID找到字符串ID
            if isinstance(number, np.ndarray) and number.shape[0] > 0:
                number = number[0]
            id2rxnclass = {v: k for k, v in rxnclass2id.items()}
            
            # 根据数字ID找到字符串ID
            rxn_str_id = id2rxnclass.get(number)
            if rxn_str_id is None:
                return f"数字ID {number} 不在映射中", None
            
            # 根据字符串ID找到类别名称
            class_name = rxnclass2name.get(rxn_str_id)
            if class_name is None:
                return rxn_str_id, "字符串ID不在映射中"
            
            return rxn_str_id, class_name
    rxn_str_id, class_name = get_class_name_from_number(predicted_class_id)
    return  predicted_class_id,rxn_str_id, class_name

def process_smiles_batch(eval_df):
    with open('rxnclass2id.json', 'r') as f:
        rxnclass2id = json.load(f)
    with open('rxnclass2name.json', 'r') as f:
        rxnclass2name = json.load(f)
    results = []
    data_to_predict = eval_df['rxn'].values.tolist()
    y_preds = model.predict(data_to_predict)
    df_preds = pd.DataFrame(y_preds)
    result = df_preds.iloc[[0]]
    result_transposed = result.transpose()
    result_transposed.reset_index(inplace=True)
    predicted_class_ids = result_transposed.iloc[:, 1]
    def get_class_name_from_number(number):
        # 反转rxnclass2id字典，以便根据数字ID找到字符串ID
        if isinstance(number, np.ndarray) and number.shape[0] > 0:
            number = number[0]
        id2rxnclass = {v: k for k, v in rxnclass2id.items()}
        
        # 根据数字ID找到字符串ID
        rxn_str_id = id2rxnclass.get(number)
        if rxn_str_id is None:
            return f"数字ID {number} 不在映射中", None
            
        # 根据字符串ID找到类别名称
        class_name = rxnclass2name.get(rxn_str_id)
        if class_name is None:
            return rxn_str_id, "字符串ID不在映射中"
        return rxn_str_id, class_name
    for predicted_class_id in predicted_class_ids:
        rxn_str_id, class_name = get_class_name_from_number(predicted_class_id)
        results.append((predicted_class_id, rxn_str_id, class_name))
    results_df = pd.DataFrame(results, columns=['Reaction Class', 'ID', 'Name'])
    return results_df






