import os
import sys
from utils import read_data, output_data
import numpy as np
from config import language_to_path
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def cal_tool_create_res_ch(results, all_create_data, key='eval'):
    format_list = []
    accuracy_list = []
    content_list = []
    executability_list = []
    richness_list = []
    overall_list = []

    res_dict = {}
    for res in results:
        data = res['data']
        step = res['step']
        eval = res['eval']
        if json.dumps(data,ensure_ascii=False) not in res_dict.keys():
            res_dict[json.dumps(data,ensure_ascii=False)] = [{"step":step, "eval":eval}]
        else:
            res_dict[json.dumps(data,ensure_ascii=False)].append({"step":step, "eval":eval})
    
    for data in all_create_data:
        tmp_format_list = []
        tmp_accuracy_list = []
        tmp_content_list = []
        tmp_executability_list = []
        tmp_richness_list = []
        tmp_overall_list = []
        if json.dumps(data,ensure_ascii=False) not in res_dict.keys():
            # empty results are scoring with 0
            for g in data['golden']:
                tmp_format_list.append(0)
                tmp_accuracy_list.append(0)
                tmp_content_list.append(0)
                tmp_executability_list.append(0)
                tmp_richness_list.append(0)
                tmp_overall_list.append(0)
        
        else:
            golden_steps = [data['step'].split(' ')[0] for data in data['golden']]
            eval_res = res_dict[json.dumps(data,ensure_ascii=False)]
            for e_r in eval_res:
                if e_r['step'].split(' ')[0] in golden_steps:
                    tmp_format_list.append(e_r['eval'][0]['格式遵从性分数'])
                    tmp_accuracy_list.append(e_r['eval'][1]['准确性分数'])
                    tmp_content_list.append(e_r['eval'][2]['内容合理性分数'])
                    tmp_executability_list.append(e_r['eval'][3]['可执行性分数'])
                    tmp_richness_list.append(e_r['eval'][4]['丰富度分数'])
                    tmp_overall_list.append(e_r['eval'][5]['总分'])
                else:
                    tmp_format_list.append(0)
                    tmp_accuracy_list.append(0)
                    tmp_content_list.append(0)
                    tmp_executability_list.append(0)
                    tmp_richness_list.append(0)
                    tmp_overall_list.append(0)

        format_list.append(tmp_format_list)
        accuracy_list.append(tmp_accuracy_list)
        content_list.append(tmp_content_list)
        executability_list.append(tmp_executability_list)
        richness_list.append(tmp_richness_list)
        overall_list.append(tmp_overall_list)
    

    avg_format = [np.mean(sublist) for sublist in format_list]
    flat_format = [item for sublist in format_list for item in sublist]
    avg_accuracy = [np.mean(sublist) for sublist in accuracy_list]
    flat_accuracy = [item for sublist in accuracy_list for item in sublist]
    avg_content = [np.mean(sublist) for sublist in content_list]
    flat_content = [item for sublist in content_list for item in sublist]
    avg_executability = [np.mean(sublist) for sublist in executability_list]
    flat_executability = [item for sublist in executability_list for item in sublist]
    avg_richness = [np.mean(sublist) for sublist in richness_list]
    flat_richness = [item for sublist in richness_list for item in sublist]
    avg_overall = [np.mean(sublist) for sublist in overall_list]
    flat_overall = [item for sublist in overall_list for item in sublist]

    
    return round(np.mean(flat_format)/10, 4), round(np.mean(flat_accuracy)/10, 4), round(np.mean(flat_content)/10, 4), round(np.mean(flat_executability)/10, 4), round(np.mean(flat_richness)/10, 4), round(np.mean(flat_overall)/10, 4)

def cal_tool_create_res_en(results, all_create_data, key='eval'):
    format_list = []
    accuracy_list = []
    content_list = []
    executability_list = []
    richness_list = []
    overall_list = []

    res_dict = {}
    for res in results:
        data = res['data']
        step = res['step']
        eval = res['eval']
        if json.dumps(data,ensure_ascii=False) not in res_dict.keys():
            res_dict[json.dumps(data,ensure_ascii=False)] = [{"step":step, "eval":eval}]
        else:
            res_dict[json.dumps(data,ensure_ascii=False)].append({"step":step, "eval":eval})
    
    for data in all_create_data:
        tmp_format_list = []
        tmp_accuracy_list = []
        tmp_content_list = []
        tmp_executability_list = []
        tmp_richness_list = []
        tmp_overall_list = []
        if json.dumps(data,ensure_ascii=False) not in res_dict.keys():
            for g in data['golden']:
                tmp_format_list.append(0)
                tmp_accuracy_list.append(0)
                tmp_content_list.append(0)
                tmp_executability_list.append(0)
                tmp_richness_list.append(0)
                tmp_overall_list.append(0)
        
        else:
            golden_steps = [data['step'].split(' ')[0] for data in data['golden']]
            eval_res = res_dict[json.dumps(data,ensure_ascii=False)]
            for e_r in eval_res:
                if e_r['step'].split(' ')[0] in golden_steps:
                    tmp_format_list.append(e_r['eval'][0]['Format Compliance Score'])
                    tmp_accuracy_list.append(e_r['eval'][1]['Accuracy Score'])
                    tmp_content_list.append(e_r['eval'][2]['Content Reasonableness Score'])
                    tmp_executability_list.append(e_r['eval'][3]['Executability Score'])
                    tmp_richness_list.append(e_r['eval'][4]['Richness Score'])
                    tmp_overall_list.append(e_r['eval'][5]['Total Score'])
                else:
                    tmp_format_list.append(0)
                    tmp_accuracy_list.append(0)
                    tmp_content_list.append(0)
                    tmp_executability_list.append(0)
                    tmp_richness_list.append(0)
                    tmp_overall_list.append(0)

        format_list.append(tmp_format_list)
        accuracy_list.append(tmp_accuracy_list)
        content_list.append(tmp_content_list)
        executability_list.append(tmp_executability_list)
        richness_list.append(tmp_richness_list)
        overall_list.append(tmp_overall_list)
    

    avg_format = [np.mean(sublist) for sublist in format_list]
    flat_format = [item for sublist in format_list for item in sublist]
    avg_accuracy = [np.mean(sublist) for sublist in accuracy_list]
    flat_accuracy = [item for sublist in accuracy_list for item in sublist]
    avg_content = [np.mean(sublist) for sublist in content_list]
    flat_content = [item for sublist in content_list for item in sublist]
    avg_executability = [np.mean(sublist) for sublist in executability_list]
    flat_executability = [item for sublist in executability_list for item in sublist]
    avg_richness = [np.mean(sublist) for sublist in richness_list]
    flat_richness = [item for sublist in richness_list for item in sublist]
    avg_overall = [np.mean(sublist) for sublist in overall_list]
    flat_overall = [item for sublist in overall_list for item in sublist]

    
    return round(np.mean(flat_format)/10, 4), round(np.mean(flat_accuracy)/10, 4), round(np.mean(flat_content)/10, 4), round(np.mean(flat_executability)/10, 4), round(np.mean(flat_richness)/10, 4), round(np.mean(flat_overall)/10, 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    print("--------- tool create evaluation results on {} ---------".format(args.language))
    for model in args.models:
        print("---- Evaluated model : {}".format(model))
        tool_eval_path = os.path.join(data_path, model, 'eval', 'tool_creation_eval.json')
        tool_eval_res = read_data(tool_eval_path)

        tool_create_path = os.path.join(data_path, model, 'eval', 'tool_creation_post_process.json')
        tool_create_res = read_data(tool_create_path)

        if args.language == 'ch':
            local_format, local_accuracy, local_content, local_executability, local_richness, local_overall = cal_tool_create_res_ch(tool_eval_res, tool_create_res)
        elif args.language == 'en':
            local_format, local_accuracy, local_content, local_executability, local_richness, local_overall = cal_tool_create_res_en(tool_eval_res, tool_create_res)
        
        print('Format Compliance Score: {}, Accuracy Score: {}, Content Reasonableness Score: {}, Executability Score: {}, Richness Score: {}, Overall Score: {}\n'.format(local_format, local_accuracy, local_content, local_executability, local_richness, local_overall))
