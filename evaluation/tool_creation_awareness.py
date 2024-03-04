import os
import sys
from utils import read_data, output_data, collect_pred_golden_for_aware, cal_acc_for_aware
import numpy as np
import argparse
from config import language_to_path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    print("--------- tool_creation_awareness evaluation results on {} ---------".format(args.language))
    for model in args.models:
        print("---- Evaluated model : {}".format(model))
        path = os.path.join(data_path, model, 'tool_creation_awareness.json')
        if model in ['gpt-3.5', 'gpt-4']:
            preds, goldens = collect_pred_golden_for_aware(path, model_type='gpt')
        else:
            preds, goldens = collect_pred_golden_for_aware(path)

        list_accuracy, element_accuracy = cal_acc_for_aware(preds, goldens)
        print("accuracy (global level): {}, accuracy (local level): {}\n".format(list_accuracy, element_accuracy))
