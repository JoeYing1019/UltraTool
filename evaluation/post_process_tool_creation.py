import json
import re
from utils import read_data
from config import language_to_path
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pattern = re.compile(r'(\[\{.*?\}\])')
def custom_serializer(obj):
    if isinstance(obj, type(Ellipsis)):
        return None  
    
def output_data(data,file_path):
    with open(file_path,'w',encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False, default=custom_serializer)+'\n')
            f.flush()
    return

def collect_pred_golden(path, model_type='other'):
    results = read_data(path)
    cnt = 0
    for idx, res in enumerate(results):
        if model_type == 'gpt':
            golden = res['data']['reference']
            pred_ = res['init output']
        else:
            golden = res['question']['reference']
            pred_ = res['total output']['init output']
        
        if type(pred_) == list:
            pred_ = pred_[0]
            
        if isinstance(pred_, list):
            pred = pred_
        elif isinstance(pred_, str):
            # general post process
            pred_text = pred_.replace('json','').strip('`').replace("输出：", "").replace("\n", "").replace("\\_", "_").replace("true", "True").replace("false", "False").replace("null", "\"null\"").replace("})", "}").replace("} }", "}}").replace("}  }", "}}").replace("}   }", "}}").replace("}    }", "}}").replace("}     }", "}}").replace("}      }", "}}").replace("}       }", "}}").replace("}        }", "}}").replace("}         }", "}}").replace("}          }", "}}").replace("}           }", "}}").replace("}            }", "}}")
            # not good start
            if not pred_text.startswith('['):
                id_ = pred_text.find("[")
                if id_ !=-1:
                    pred_text = pred_text[id_:]
            # not good end
            if not pred_text.endswith(']'):
                id_ = pred_text.rfind("]")
                if id_ !=-1:
                    pred_text = pred_text[:id_+1]
                elif '[' in pred_text:
                    idx_ = pred_text.rfind('}}}},')
                    if idx_ !=-1 :
                        pred_text = pred_text[:idx_+4] +']'
                    else:
                        pred_text = pred_text + "]"
                    
            # multiple list
            if model_type != 'gpt' and pred_text.count(']') > 1:
                matches = pattern.findall(pred_text)
                if len(matches) > 0:
                    pred_text = matches[-1] # choose the last list
                else:
                    id_ = pred_text.find("]")
                    if id_ != -1:
                        pred_text = pred_text[:id_+1]
                        
            successful_parse = False

            try:
                pred = eval(pred_text)
                successful_parse = True
            except Exception:
                pass
            
            # lack }
            try:
                if not successful_parse:
                    pred_text_ = pred_text[:-1] + '}' + pred_text[-1]
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try: # lack }}
                    if not successful_parse:
                        pred_text_ = pred_text_[:-1] + '}' + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    try: # lack }}}
                        if not successful_parse:
                            pred_text_ = pred_text_[:-1] + '}' + pred_text_[-1]
                            pred = eval(pred_text_)
                            successful_parse = True
                    except Exception:
                        try: # lack }}}}
                            if not successful_parse:
                                pred_text_ = pred_text_[:-1] + '}' + pred_text_[-1]
                                pred = eval(pred_text_)
                                successful_parse = True
                        except Exception:
                            pass
            try: # lack "}
                if not successful_parse:
                    pred_text_ = pred_text[:-1] + '\"}' + pred_text[-1]
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try: # lack "}}
                    if not successful_parse:
                        pred_text_ = pred_text_[:-1] + '\"}}' + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    try: # lack "}}}
                        if not successful_parse:
                            pred_text_ = pred_text_[:-1] + '\"}}}' + pred_text_[-1]
                            pred = eval(pred_text_)
                            successful_parse = True
                    except Exception:
                        pass
            
            # redundant }
            try:
                if not successful_parse:
                    pred_text_ = pred_text[:-2] + pred_text[-1]
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try:# redundant }}
                    if not successful_parse:
                        pred_text_ = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass
                
            try:
                if not successful_parse:
                    id_ = pred_text.find("}}}}}}}}}")
                    if id_ != -1:
                        pred_text_ = pred_text[:id_] + pred_text[id_+2:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_ = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass

            try:
                if not successful_parse:
                    id_ = pred_text.find("}}}}}}}")
                    if id_ != -1:
                        pred_text_ = pred_text[:id_] + pred_text[id_+1:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_ = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass

            try:
                if not successful_parse:
                    id_ = pred_text.find("}}}}}}")
                    if id_ != -1:
                        pred_text_ = pred_text[:id_] + pred_text[id_+1:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_ = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass

            try:
                if not successful_parse:
                    id_ = pred_text.find("}}}}")
                    if id_ != -1:
                        pred_text_ = pred_text[:id_] + '}' + pred_text[id_:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_ = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass
            # redundant ]
            try:
                if not successful_parse:
                    if pred_text[-3] == ']':
                        pred_text_ = pred_text[:-3] + pred_text[-2:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                pass

            # the last step is incomplete, just get the previous one
            try:
                if not successful_parse:
                    idx_ = pred_text.rfind('}}},')
                    if idx_ !=-1 and '[' in pred_text:
                        pred_text_ = pred_text[:idx_+3] +']'
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                pass
            
            try:
                if not successful_parse:
                    id_ = pred_text.find("}}}}, ")
                    if id_ != -1 and pred_text[id_+6] != '{':
                        pred_text_ = pred_text[:id_+6] + '{' + pred_text[id_+6:]
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                pass

            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}}", "}}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}}}", "}}}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}}}", "}}}}}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}}}}", "}}}}}}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_2 = pred_text_[:-1] + '}' + pred_text_[-1]
                        pred = eval(pred_text_2)
                        successful_parse = True
                except Exception:
                    pass

            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}", "}}}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass
            
            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("}}}}", "}}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass
            # not good string format
            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("{'", '{"').replace("'}", '"}').replace("', '", '", "').replace("\", '", "\", \"").replace("', \"", "\", \"").replace("': '", '": "').replace("': \"", "\": \"").replace("\": '", "\": \"").replace("': {", "\": {").replace("}, \'", "}, \"").replace("\': ", "\": ").replace(", \'", ", \"").replace("\':", "\":").replace("}, \"step", "}, {\"step")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_2 = pred_text_.replace("}}}}", "}}}")
                        pred = eval(pred_text_2)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_3 = pred_text_[:-2] + pred_text_[-1]
                        pred = eval(pred_text_3)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_4 = pred_text_.replace("}}}}}}", "}}}}}")
                        pred = eval(pred_text_4)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        if pred_text_[-3] == ']':
                            pred_text_5 = pred_text_[:-3] + pred_text_[-2:]
                            pred = eval(pred_text_5)
                            successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_6 = pred_text_[:-2]
                        pred = eval(pred_text_6)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_7 = pred_text_.replace("}}}}", "}}}}}")
                        pred = eval(pred_text_7)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_8 = pred_text_[:-1] + '}' + pred_text_[-1]
                        pred = eval(pred_text_8)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        pred_text_9 = pred_text_.replace("}}}}}}", "}}}}}}}}")
                        pred = eval(pred_text_9)
                        successful_parse = True
                except Exception:
                    pass
                try:
                    if not successful_parse:
                        idx_ = pred_text_.rfind('}}},')
                        if idx_ !=-1 and '[' in pred_text_:
                            pred_text_10 = pred_text_[:idx_+3] +']'
                            pred = eval(pred_text_10)
                            successful_parse = True
                except Exception:
                    pass
            
            if not successful_parse: 
                cnt+=1
                pred = []

        else:
            print()
        results[idx]['golden'] = golden
        results[idx]['pred'] = pred

    print('number of bad json format: {}'.format(cnt))
    return results



def collect_pred_res(results):
    for idx, res in enumerate(results):
        pred = res['pred']
        golden = res['golden']
        p_value = []
        for g in golden:
            g_name = g['step']
            flag = False
            if pred == Ellipsis:
                continue
            for p in pred:
                    try:
                        if type(p) == list:
                            p == p[0]
                        p_name = p['step']
                        if p_name.split(' ')[0] == g_name.split(' ')[0]: # match step number
                            p_value.append({'step': p['step'], 'tool':p['tool']})
                            flag = True
                            break
                    except Exception:
                        pass
            if not flag:
                p_value.append({'step': g['step'], 'tool':''})
        results[idx]["pred_need_eval"]=p_value
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=list, default=['gpt-3.5', 'gpt-4', 'chatglm3-6b', 'llama-7b', 'vicuna-7b', 'baichuan-7b', 'qianwen-7b', 'mistral-7b', 'llama-13b', 'vicuna-13b', 'baichuan-13b', 'qianwen-14b', 'llama-70b', 'qianwen-72b'], help="model name of evaluated models, e.g. gpt-3.5 ,gpt-4, vicuna-7b")
    parser.add_argument("--language", type=str, choices=['ch', 'en'], help="evaluation language, en (English) or ch(Chinese)")
    args = parser.parse_args()
    
    data_path = language_to_path[args.language]

    for model in args.models:
        print("---- Post-precessed model : {}".format(model))
        path = os.path.join(data_path, model, 'tool_creation.json')

        if model in ['gpt-3.5', 'gpt-4']:
            results = collect_pred_golden(path, model_type='gpt')
        else:
            results = collect_pred_golden(path)

        out_results = collect_pred_res(results)
        out_dir = os.path.join(data_path, model, "eval")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "tool_creation_post_process.json")     

        output_data(out_results, out_path)