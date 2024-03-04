import json
import numpy as np
import re
from sklearn.metrics import accuracy_score
from jiwer import wer
from typing import List, Union
from textdistance.algorithms.edit_based import levenshtein
pattern = re.compile(r'(\[\{.*?\}\])')

class Distance:
    """
    An abstract class representing a distance metric. Every distance is normalized to be defined in [0, 1].
    """
    def __call__(self,
                 truth: Union[str, List[str]],
                 hypothesis: Union[str, List[str]]) -> float:
        """
        Return the distance between truth and hypothesis.

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The distance value between `truth` and `hypothesis`.
        """
        raise NotImplementedError

    @staticmethod
    def get_instance(distance: str) -> 'Distance':
        """
        This static method allows to build a Distance object.

        :param distance: The distance to be returned.
        :return: A `Distance` object to evaluate the distance between two texts.
        """
        # assert distance in DISTANCE_OPTIONS, "Allowed distances: {}".format(DISTANCE_OPTIONS)
        if distance == "word":
            return WordDistance()
        if distance == "char":
            return CharDistance()

class WordDistance(Distance):
    """
    The Word-level distance, implemented through the Word Error Rate (WER).
    """
    def __call__(self, truth: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
        """
        Evaluates the word-level distance

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The word-level distance.
        """
        return wer(truth=truth, hypothesis=hypothesis)


class CharDistance(Distance):
    """
    The Character-level distance, implemented through the normalised Levenshtein distance.
    """
    def __call__(self, truth: Union[str, List[str]], hypothesis: Union[str, List[str]]) -> float:
        """
        Evaluates the character-level distance

        :param truth: The ground-truth sentence as a string or list of words.
        :param hypothesis: The hypothesis sentence as a string or list of words.
        :return: The character-level distance.
        """
        return levenshtein.normalized_distance(truth, hypothesis)


def output_data(data,file_path):
    with open(file_path,'w',encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')
            f.flush()
    return 

def read_data(file_path):
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            data.append(line)
    return data

def collect_pred_golden_for_aware(path, model_type='other'):
    results = read_data(path)
    preds = []
    goldens = []
    cnt = 0
    for res in results:
        if model_type == 'gpt':
            golden = res['data']['reference']
            pred_ = res['init output']
        else:
            golden = res['question']['reference']
            pred_ = res['total output']['init output'][0]
        if isinstance(pred_, list):
            if not isinstance(pred_[0], str):
                pred = pred_
            else:
                pred_ = pred_[0]
        if isinstance(pred_, str):
            # general process
            pred_text = pred_.replace('json','').strip('`').replace("输出：", "").replace("\n", "").replace("\\_", "_").replace("true", "True").replace("false", "False").replace("null", "\"null\"").replace("None", "\"None\"").replace("})", "}").replace("，", ",").replace("}, \"step", "}, {\"step").replace("# ...", "").replace("} }", "}}").replace("}  }", "}}").replace("}   }", "}}").replace("}    }", "}}").replace("}     }", "}}").replace("}      }", "}}").replace("}       }", "}}").replace("}        }", "}}").replace("}         }", "}}").replace("}          }", "}}").replace("}           }", "}}").replace("}            }", "}}")
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
                    idx_ = pred_text.rfind('},')
                    if idx_ !=-1 :
                        pred_text = pred_text[:idx_+1] +']'
                    else:
                        pred_text = pred_text + "]"
            successful_parse = False

             # multiple list
            if model_type != 'gpt' and pred_text.count(']') > 1:
                matches = pattern.findall(pred_text)
                if len(matches) > 0:
                    pred_text = matches[-1] # choose the last list
                else:
                    id_ = pred_text.find("]")
                    if id_ != -1:
                        pred_text = pred_text[:id_+1]

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
                try:# lack }
                    if not successful_parse:
                        pred_text_ = pred_text[:-1] + '"}' + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass
            
            # the last step is incomplete, just get the previous one 
            try:
                if not successful_parse:
                    idx_ = pred_text.rfind('},')
                    if idx_ !=-1 and '[' in pred_text:
                        pred_text_ = pred_text[:idx_+1] +']'
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                pass

            # not good string format
            try:
                if pred_text.strip() =='format' or not successful_parse:
                    pred_text_ = pred_text.replace("{'", '{"').replace("'}", '"}').replace("', '", '", "').replace("\", '", "\", \"").replace("', \"", "\", \"").replace("': '", '": "').replace("': \"", "\": \"").replace("\": '", "\": \"").replace("': {", "\": {").replace("[\'", "[").replace("\']", "]").replace(", \'", ", \"").replace("\'),", ")\",").replace("\"\"None\"\"", "\"None\"").replace("\"{step", "{\"step").replace("\')}", ")\"}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                if '[' in pred_text_:
                    pass

        else:
            pass
        
        if not successful_parse: 
            cnt+=1
            pred = []

        preds.append(pred)
        goldens.append(golden)
    print('number of bad json format: {}'.format(cnt))
    return preds, goldens

def collect_pred_golden_for_param(path, model_type='other'):
    results = read_data(path)
    preds = []
    goldens = []
    cnt = 0
    for res in results:
        if model_type == 'gpt':
            golden = res['data']['reference']
            pred_ = res['init output']
        else:
            golden = res['question']['reference']
            pred_ = res['total output']['init output'][0]
        if isinstance(pred_, list):
            if not isinstance(pred_[0], str):
                pred = pred_
            else:
                pred_ = pred_[0]
        if isinstance(pred_, str):
            # general process
            pred_text = pred_.replace('json','').strip('`').replace("输出：", "").replace("\n", "").replace("\\_", "_").replace("true", "True").replace("false", "False").replace("null", "\"null\"").replace("None", "\"None\"").replace("})", "}").replace("，", ",").replace("}, \"step", "}, {\"step").replace("# ...", "").replace("}{", "}, {").replace("} }", "}}").replace("}  }", "}}").replace("}   }", "}}").replace("}    }", "}}").replace("}     }", "}}").replace("}      }", "}}").replace("}       }", "}}").replace("}        }", "}}").replace("}         }", "}}").replace("}          }", "}}").replace("}           }", "}}").replace("}            }", "}}")
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
                    idx_ = pred_text.rfind('}},')
                    if idx_ !=-1 :
                        pred_text = pred_text[:idx_+2] +']'
                    else:
                        pred_text = pred_text + "]"
            successful_parse = False

             # multiple list
            if model_type != 'gpt' and pred_text.count(']') > 1:
                matches = pattern.findall(pred_text)
                if len(matches) > 0:
                    pred_text = matches[-1] # choose the last list
                else:
                    id_ = pred_text.find("]")
                    if id_ != -1:
                        pred_text = pred_text[:id_+1]

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
                try:# lack }}
                    if not successful_parse:
                        pred_text_ = pred_text_[:-1] + '}' + pred_text_[-1]
                        pred = eval(pred_text_)
                        successful_parse = True
                except Exception:
                    pass
            
            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("\"}, {\"step", "\"}}, {\"step")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            try:# lack "}
                if not successful_parse:
                    pred_text_ = pred_text[:-1] + '"}' + pred_text[-1]
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            try:# lack ]}}
                if not successful_parse:
                    pred_text_ = pred_text[:-1] + ']}}' + pred_text[-1]
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
                    pred_text_ = pred_text.replace("}}}", "}}")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                pass

            # the last step is incomplete, just get the previous one
            try:
                if not successful_parse:
                    idx_ = pred_text.rfind('}},')
                    if idx_ !=-1 and '[' in pred_text:
                        pred_text_ = pred_text[:idx_+2] +']'
                        pred = eval(pred_text_)
                        successful_parse = True
            except Exception:
                pred_text_ = pred_text.replace("{'", '{"').replace("'}", '"}').replace("', '", '", "').replace("\", '", "\", \"").replace("', \"", "\", \"").replace("': '", '": "').replace("': \"", "\": \"").replace("\": '", "\": \"").replace("': {", "\": {").replace("[\'", "[").replace("\']", "]").replace(", \'", ", \"").replace("\'),", ")\",").replace("\': ", "\": ").replace("\', {", "\", {").replace("\')}", ")\"}").replace("}},\" {", "}}, {")
                pass

            # not good string forma
            try:
                if not successful_parse:
                    pred_text_ = pred_text.replace("{'", '{"').replace("'}", '"}').replace("', '", '", "').replace("\", '", "\", \"").replace("', \"", "\", \"").replace("': '", '": "').replace("': \"", "\": \"").replace("\": '", "\": \"").replace("': {", "\": {").replace("[\'", "[").replace("\']", "]").replace(", \'", ", \"").replace("\'),", ")\",").replace("\': ", "\": ").replace("\', {", "\", {").replace("\')}", ")\"}").replace("}},\" {", "}}, {").replace("\"\"None\"\"", "\"None\"")
                    pred = eval(pred_text_)
                    successful_parse = True
            except Exception:
                try:
                    if not successful_parse:
                        pred_text_2 = pred_text_.replace("}}}", "}}")
                        pred = eval(pred_text_2)
                        successful_parse = True
                except Exception:
                    pass
                try:# lack ]}}
                    if not successful_parse:
                        pred_text_3 = pred_text_[:-1] + ']}}' + pred_text_[-1]
                        pred = eval(pred_text_3)
                        successful_parse = True
                except Exception:
                    pass
        else:
            pass
        
        if not successful_parse: 
            cnt+=1
            pred = []

        preds.append(pred)
        goldens.append(golden)
    print('number of bad json format: {}'.format(cnt))
    return preds, goldens

def cal_acc_for_aware(preds, goldens):
    p_values = []
    g_values = []
    for pred, golden in zip(preds, goldens):
        p_value = []
        g_value = []
        for g in golden:
            g_name = g['step']
            g_value.append(int(g['tool']))
            flag = False
            if type(pred) == list:
                for p in pred:
                    try:
                        p_name = p['step']
                        if p_name.split(' ')[0] == g_name.split(' ')[0]:
                            p_value.append(int(p['tool']))
                            flag = True
                            break
                    except Exception:
                        pass
            if not flag:
                if int(g['tool']) == 1:
                    error_value = 0 # if not found, construct as error prediction
                else:
                    error_value = 1
                p_value.append(error_value)
        p_values.append(p_value)
        g_values.append(g_value)
    list_accuracy = sum([i == j for i, j in zip(p_values, g_values)]) / len(g_values)

    flat_list1 = [item for sublist in p_values for item in sublist]
    flat_list2 = [item for sublist in g_values for item in sublist]
    element_accuracy = accuracy_score(flat_list1, flat_list2)
    
    return round(list_accuracy, 4), round(element_accuracy, 4)


def cal_acc_for_select(preds, goldens):
    p_values = []
    g_values = []
    for pred, golden in zip(preds, goldens):
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], list):
            pred = pred[0]

        p_value = []
        g_value = []
        for g in golden:
            g_name = g['step']
            g_value.append(g['tool'])
            flag = False
            for p in pred:
                try:
                    p_name = p['step']
                    if p_name.split(' ')[0] == g_name.split(' ')[0]:
                        if "\\_" in str(p['tool']):
                            p['tool'] = p['tool'].replace("\\_", '_')
                        p_value.append(p['tool'])
                        flag = True
                        break
                except Exception:
                    pass
            if not flag:
                p_value.append("no_good")
        p_values.append(p_value)
        g_values.append(g_value)
    list_accuracy = sum([i == j for i, j in zip(p_values, g_values)]) / len(g_values)
    flat_list1 = [item for sublist in p_values for item in sublist]
    flat_list2 = [item for sublist in g_values for item in sublist]
    element_accuracy = accuracy_score(flat_list1, flat_list2)
    
    return round(list_accuracy, 4), round(element_accuracy, 4)

def cal_param_res(preds, goldens):
    distance_char = Distance.get_instance(distance='char')
    res_list = []
    for pred, golden in zip(preds, goldens):
        if isinstance(pred, list) and len(pred) >0 and isinstance(pred[0], list): # TODO: 有多个的情况
            pred = pred[0]
        tmp_res = []
        for g in golden:
            g_name = g['step']
            g_params = g['param']
            flag_step = False

            for p in pred:
                if p == Ellipsis or type(p) != dict or len(p) == 0  or 'step' not in p.keys() or 'param' not in p.keys():
                    continue
                p_name = p['step']
                if type(p_name) == str and p_name.split(' ')[0] == g_name.split(' ')[0]:
                    flag_step = True
                    p_params = p['param']
                    tmp_score = []
                    for g_key, g_value in g_params.items():
                        flag_key = False
                        if  not isinstance(p_params, dict):
                            continue
                        for p_key, p_value in p_params.items():
                            if isinstance(p_key, str) and "\\_" in p_key:
                                p_key = p_key.replace("\\_", '_')
                            if isinstance(p_value, str) and "\\_" in p_value:
                                p_value = p_value.replace("\\_", '_')
                            if p_key == g_key:
                                flag_key = True
                                
                                if len(g_value) == 0:
                                    if len(p_value) ==0:
                                        score = 1.
                                    else:
                                        score = 0.
                                else:
                                    score = 1 - distance_char(str(g_value), str(p_value))

                                if isinstance(score, int):
                                    score = float(score)
                                tmp_score.append(score)
                                break
                        if not flag_key:
                            tmp_score.append(0.)
                    if len(tmp_score) == 0:
                        tmp_res.append(0.)
                    else:
                        tmp_res.append(np.mean(tmp_score))
                    break
            if not flag_step:
                tmp_res.append(0.)
        if len(tmp_res) == 0:
            res_list.append([0.])
        else:
            res_list.append(tmp_res)

    avg_res = [np.mean(sublist) for sublist in res_list] 
    flat_res = [item for sublist in res_list for item in sublist]

    return round(np.mean(flat_res), 4)  