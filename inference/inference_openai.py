import os
import sys
from openai_api import multi_threading_gpt4_infer
from utils import read_data, output_data, generate_prompt_chinese, generate_prompt_english
import json
import openai
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

language_to_path = {
    "ch" : "data/Chinese-dataset",
    "en" : "data/English-dataset"
}

language_to_out_path = {
    "ch" : "predictions/Chinese-dataset",
    "en" : "predictions/English-dataset"
}

language_to_prompt_func = {
    "ch" : generate_prompt_chinese,
    "en" : generate_prompt_english
}

name_to_model = {
    "gpt-3.5" : "gpt-3.5-turbo-1106",
    "gpt-4" : "gpt-4-1106-preview"
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['gpt-3.5', 'gpt-4'], help="model name of openai models, e.g. gpt-3.5 or gpt-4")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    parser.add_argument("--tasks", type=list, default=['planning', 'tool_usage_awareness', 'tool_creation', 'tool_usage', 'tool_creation_awareness', 'tool_selection'], help="tasks list for evaluatopn from UltraTool")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    output_path = language_to_out_path[args.language]
    prompt_func = language_to_prompt_func[args.language]
    model_name = name_to_model[args.model]


    for task in args.tasks:
        prompt_data = []
        out_datas = []
        task_path = os.path.join(data_path, "test_set","{}.json".format(task))
        datas = read_data(task_path)

        example_path = os.path.join(data_path, "example","{}.json".format(task))
        example = json.load(open(example_path, 'r',encoding='utf-8'))
        
        for data in datas:
            warp_prompt = prompt_func('few_shot_cot', task, data, example)
            prompt_data.append({"prompt_new": warp_prompt, "ori_data": data})
        
        out_dir = os.path.join(output_path, args.model)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "{}.json".format(task))        

        tmp_out_dir = os.path.join(output_path, args.model, "inferback")
        if not os.path.exists(tmp_out_dir):
            os.makedirs(tmp_out_dir)
        tmp_out_path = os.path.join(tmp_out_dir, "inferback_{}.json".format(task))

        outf = open(tmp_out_path,'w',encoding='utf8')
        results = multi_threading_gpt4_infer(prompt_data, None, outf, model_name=model_name, max_workers=50)
    
        for res in results:
            gen_res = res['model_response']
            data = res['ori_data']
            new_data = {'data': data, 'init output': gen_res}
            out_datas.append(new_data)

        output_data(out_datas, out_path)






