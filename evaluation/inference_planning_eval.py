import json
import os
import openai
from tqdm import tqdm
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.openai_api import multi_threading_gpt4_infer
from config import language_to_path, language_to_plan_eval_prompt
from utils import read_data, output_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    prompt = language_to_plan_eval_prompt[args.language]

    for model in args.models:
        print("---- Evaluated model : {}".format(model))
        prompt_data = []
        out_datas = []
        path = os.path.join(data_path, model, 'planning.json')
        datas = read_data(path)
        for res in datas:
            if model in ['gpt-3.5', 'gpt-4']:
                question = res['data']['input']
                reference = res['data']['reference']
                answer = res['init output']
            else:
                question = res['question']['input']
                reference = res['question']['reference']
                answer = res['total output']['init output']

            if type(answer) == list:
                answer = answer[0]
            # remove the redundant generation tokens
            answer = answer.replace("<|im_end|>", "").replace("<|im_end|", "").replace("<|im_start|>", "").replace("<|im_start|", "")
            if not answer.strip().startswith("1. "):
                # remove redundant generation results at start
                id_ = answer.find("1. ")
                if id_ != -1:
                    answer = answer[id_:]
                # remove redundant generation results at end
                if '\n\n' in answer and answer.split('\n\n')[-1]!= '' and answer.split('\n\n')[-1][0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    answer = '\n\n'.join(answer.split('\n\n')[:-1])

            warp_prompt = prompt.format(question=question, reference=reference, answer=answer)
            prompt_data.append({"prompt_new": warp_prompt, "ori_data": res})

        out_dir = os.path.join(data_path, model, "eval")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "planning_eval.json")        

        tmp_out_dir = os.path.join(data_path, model, "eval", "inferback")
        if not os.path.exists(tmp_out_dir):
            os.makedirs(tmp_out_dir)
        tmp_out_path = os.path.join(tmp_out_dir, "inferback_planning_eval.json")

        outf = open(tmp_out_path,'w',encoding='utf8')
        results = multi_threading_gpt4_infer(prompt_data, None, outf, model_name="gpt-4-1106-preview", max_workers=50)

        for res in results:
            ans = json.loads(res['model_response'].replace('json','').strip('`'))
            data = res['ori_data']
            new_data = {'data': data, 'eval': ans}
            out_datas.append(new_data)

        output_data(out_datas, out_path)







