"""Evaluation metrics and utilities."""

import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Any


def get_performance(actual: List[int], preds: List[int], dict_mapping: Dict[str, int]):
    """Print performance metrics for a single task."""
    print(classification_report(actual, preds))
    print('--' * 10)
    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(actual, preds)))
    print('--' * 10)
    print('Actual counter:', Counter(actual))
    print('Prediction counter:', Counter(preds))
    print('Mapping:', dict_mapping)


def evaluate_all_tasks(eval_results: Dict[str, List], config: Dict[str, Any]):
    """Evaluate performance for all tasks."""
    tasks = ['sa', 'si', 'rel']
    task_names = ['Suicide Attempt', 'Suicide Ideation', 'Relevance']
    
    for task, task_name in zip(tasks, task_names):
        print(f'\n{"=" * 40}')
        print(f'{task_name.upper()} PERFORMANCE')
        print(f'{"=" * 40}')
        
        actual_key = f'{task}_actual'
        preds_key = f'{task}_preds'
        
        if task == 'sa':
            mapping = config['tasks']['suicide_attempt']['label_mapping']
        elif task == 'si':
            mapping = config['tasks']['suicide_ideation']['label_mapping']
        else:
            mapping = config['tasks']['relevance']['label_mapping']
            
        get_performance(
            actual=eval_results[actual_key],
            preds=eval_results[preds_key],
            dict_mapping=mapping
        )


def calculate_document_level_performance(eval_results: Dict[str, List], 
                                       test_data: List[Dict], 
                                       config: Dict[str, Any]):
    """Calculate document-level performance by aggregating paragraph predictions."""
    
    # Group predictions by document (patient)
    doc_predictions = {}
    
    for idx, sample in enumerate(test_data):
        pid = sample['pid'].split('_st_')[0]  # Extract patient ID
        
        if pid not in doc_predictions:
            doc_predictions[pid] = {
                'sa_actual': [],
                'sa_preds': [],
                'si_actual': [],
                'si_preds': []
            }
        
        doc_predictions[pid]['sa_actual'].append(eval_results['sa_actual'][idx])
        doc_predictions[pid]['sa_preds'].append(eval_results['sa_preds'][idx])
        doc_predictions[pid]['si_actual'].append(eval_results['si_actual'][idx])
        doc_predictions[pid]['si_preds'].append(eval_results['si_preds'][idx])
    
    # Aggregate to document level
    doc_actual_sa = []
    doc_pred_sa = []
    doc_actual_si = []
    doc_pred_si = []
    
    for pid in doc_predictions:
        # For suicide attempt: positive if any paragraph is positive
        doc_actual_sa.append(get_one_label_suicide_attempt(doc_predictions[pid]['sa_actual']))
        doc_pred_sa.append(get_one_label_suicide_attempt(doc_predictions[pid]['sa_preds']))
        
        # For suicide ideation: positive if any paragraph is positive
        doc_actual_si.append(get_one_label_suicide_ideation(doc_predictions[pid]['si_actual']))
        doc_pred_si.append(get_one_label_suicide_ideation(doc_predictions[pid]['si_preds']))
    
    print(f'\n{"=" * 40}')
    print('DOCUMENT-LEVEL PERFORMANCE')
    print(f'{"=" * 40}')
    
    print('\nSuicide Attempt (Document Level):')
    get_performance(doc_actual_sa, doc_pred_sa, config['tasks']['suicide_attempt']['label_mapping'])
    
    print('\nSuicide Ideation (Document Level):')
    get_performance(doc_actual_si, doc_pred_si, config['tasks']['suicide_ideation']['label_mapping'])


def get_one_label_suicide_attempt(label_list: List[int]) -> int:
    """Aggregate paragraph-level SA labels to document level."""
    if 1 in label_list:  # pos
        return 1
    elif 2 in label_list:  # neg/unsure
        return 2
    return 0  # neutral


def get_one_label_suicide_ideation(label_list: List[int]) -> int:
    """Aggregate paragraph-level SI labels to document level."""
    if 0 in label_list:  # present
        return 0
    elif 2 in label_list:  # n/a/absent
        return 2
    return 1  # neutral
