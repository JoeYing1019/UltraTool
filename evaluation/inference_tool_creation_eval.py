import json
import os
import openai
from tqdm import tqdm
from config import language_to_path, language_to_tool_eval_prompt
from utils import read_data, output_data
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.openai_api import multi_threading_gpt4_infer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]
    prompt = language_to_tool_eval_prompt[args.language]

    for model in args.models:
        print("---- Evaluated model : {}".format(model))
        prompt_data = []
        out_datas = []
        path = os.path.join(data_path, model, 'eval', 'tool_creation_post_process.json')
        datas = read_data(path)

        for res in datas:
            goldens = res['golden']
            pred_need_evals = res['pred_need_eval']
            assert len(goldens) == len(pred_need_evals)
            for golden, pred_need_eval in zip(goldens, pred_need_evals):
                question = golden['step'].split(' ')[1]
                assert len(question) > 0
                reference = golden['tool']
                answer = pred_need_eval['tool']
                if answer is None or answer == "" or answer == {} or answer == '' or len(answer) == 0:
                    continue
                if isinstance(answer, dict):
                    assert len(answer.keys()) > 0
                    warp_prompt = prompt.format(question=question, reference=reference, answer=answer)
                    prompt_data.append({"prompt_new": warp_prompt, "ori_data": res, "step": golden['step']})
        
        out_dir = os.path.join(data_path, model, "eval")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "tool_creation_eval.json")        

        tmp_out_dir = os.path.join(data_path, model, "eval", "inferback")
        if not os.path.exists(tmp_out_dir):
            os.makedirs(tmp_out_dir)
        tmp_out_path = os.path.join(tmp_out_dir, "inferback_tool_creation_eval.json")
        
        outf = open(tmp_out_path,'w',encoding='utf8')
        results = multi_threading_gpt4_infer(prompt_data,None,outf, model_name="gpt-4-1106-preview", max_workers=50)

        for res in results:
            ans = json.loads(res['model_response'].replace('json','').strip('`'))
            data = res['ori_data']
            step = res['step']
            new_data = {'data': data, "step": step, 'eval': ans}
            out_datas.append(new_data)

        output_data(out_datas, out_path)







