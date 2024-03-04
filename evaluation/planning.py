import os
import sys
from utils import read_data, output_data
import numpy as np
from config import language_to_path
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def cal_plan_res_ch(results, key='eval'):
    accuracy_list = []
    completeness_list = []
    executability_list = []
    syntactic_list = []
    structural_list = []
    efficiency_list = []
    overall_list = []
    for res in results:
        eval_res = res[key]
        accuracy_list.append(eval_res[0]['准确性分数'])
        completeness_list.append(eval_res[1]['完整性分数'])
        executability_list.append(eval_res[2]['可执行性分数'])
        syntactic_list.append(eval_res[3]['语法健全性分数'])
        structural_list.append(eval_res[4]['结构合理性分数'])
        efficiency_list.append(eval_res[5]['高效性分数'])
        overall_list.append(eval_res[6]['总分'])
    return round(np.mean(accuracy_list)/10, 4), round(np.mean(completeness_list)/10, 4), round(np.mean(executability_list)/10, 4), round(np.mean(syntactic_list)/10, 4), round(np.mean(structural_list)/10, 4), round(np.mean(efficiency_list)/10, 4), round(np.mean(overall_list)/10, 4)

def cal_plan_res_en(results, key='eval'):
    accuracy_list = []
    completeness_list = []
    executability_list = []
    syntactic_list = []
    structural_list = []
    efficiency_list = []
    overall_list = []
    for res in results:
        eval_res = res[key]
        accuracy_list.append(eval_res[0]['Accuracy Score'])
        completeness_list.append(eval_res[1]['Completeness Score'])
        executability_list.append(eval_res[2]['Executability Score'])
        syntactic_list.append(eval_res[3]['Syntactic Soundness Score'])
        structural_list.append(eval_res[4]['Structural Rationality Score'])
        efficiency_list.append(eval_res[5]['Efficiency Score'])
        overall_list.append(eval_res[6]['Overall Score'])
    return round(np.mean(accuracy_list)/10, 4), round(np.mean(completeness_list)/10, 4), round(np.mean(executability_list)/10, 4), round(np.mean(syntactic_list)/10, 4), round(np.mean(structural_list)/10, 4), round(np.mean(efficiency_list)/10, 4), round(np.mean(overall_list)/10, 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    print("--------- planning evaluation results on {} ---------".format(args.language))
    for model in args.models:
        print("---- Evaluated model : {}".format(model))
        plan_eval_path = os.path.join(data_path, model, 'eval', 'planning_eval.json')
        plan_eval_res = read_data(plan_eval_path)
        if args.language == 'ch':
            accuracy, completeness, executability, syntactic, structural, efficiency, overall=cal_plan_res_ch(plan_eval_res)
        elif args.language == 'en':
            accuracy, completeness, executability, syntactic, structural, efficiency, overall=cal_plan_res_en(plan_eval_res)
        
        print('Accuracy Score: {}, Completeness Score: {}, Executability Score: {}, Syntactic Soundness Score: {}, Structural Rationality Score: {}, Efficiency Score: {}, Overall Score: {}\n'.format(accuracy, completeness, executability, syntactic, structural, efficiency, overall))
