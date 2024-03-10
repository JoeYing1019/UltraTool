import argparse
from tqdm import tqdm
import json
import os
import numpy as np

from conversation import Conversation, SeparatorStyle

conv = Conversation(
    name="vicuna_v1.1",
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))


def parse_arguments():
    USER_PROMPT = {
        'plan': "你是一个专业的任务规划助手，给你一个用户问题，你的任务是充分理解用户问题，并制定一个合理的、可执行的多步计划以完成用户的任务。具体而言，你制订的计划应像一棵树一样分为多个子任务，输出格式示例如下：1. 任务1 \n 1.1 任务1.1 \n 1.2 任务1.2 \n 1.2.1 任务1.2.1 \n ... \n 2. 任务2 \n ...\n",
        'tool_use_awareness': '''你是一个专业的人工智能助手，给你一个任务规划，其中有些步骤带有"tool"字段，你的任务是判断完成这些步骤是否需要使用工具，如果需要使用工具则"tool"字段为"1"，否则为"0"，只需要输出有"tool"字段的步骤，输出格式示例如下：[{"step": "1.1 步骤1.1", "tool": "0"}, {"step": "2.3 步骤2.3", "tool": "1"}, ...]\n''',
        'tool_selection': '''你是一个专业的工具选择助手，给你一个任务规划和相应的工具集，其中有些步骤带有"tool"字段，你的任务是从给定的工具集中为这些步骤选择合适的工具，并在"tool"字段填入选中工具的名称（工具的"name"字段），只需要输出有"tool"字段的步骤，输出格式示例如下：[{"step": "2.1 步骤2.1", "tool": "工具名1"}, {"step": "2.3 步骤2.3", "tool": "工具名2"}, ...]\n''',
        'tool_creation_awareness': '''你是一个专业的人工智能助手，给你一个任务规划和相应的工具集，其中有些步骤带有"tool"字段，你的任务是判断这些步骤能否在给定的工具集中找到合适的工具，如果工具集中没有合适的工具（需要创造新的工具）则"tool"字段为"1"，否则为"0"，只需要输出有"tool"字段的步骤，输出格式示例如下：[{"step": "2.1 步骤2.1", "tool": "1"}, {"step": "4.2 步骤4.2", "tool": "0"}, ...]\n''',
        'tool_creation': '''你是一个专业的工具创造助手，给你一个任务规划和一个工具集，其中有些步骤带有"tool"字段，你的任务是参考给定的工具集中的工具格式，为这些步骤创造对应的工具，并在"tool"字段填入所创造的工具（json格式），只需要输出有"tool"字段的步骤，输出格式示例如下：[{"step": "2.3 步骤2.3", "tool": {"name": "工具名", "description": "工具描述", "arguments": ...}}, ...]\n''',
        'arguments_filling': '''你是一个专业的工具参数填充助手，给你用户问题，问题对应的任务规划和相应的工具集，其中有些步骤带有"tool"字段（工具名）和"param"字段，你的任务是先通过"tool"字段在工具集中找到对应工具，然后为这些步骤的"param"字段填入调用对应工具所需要的参数，参数格式为"参数名=参数值"（多个参数用','分隔），参数名来自于对应工具的"arguments"中的"properties"字段（不需要全部使用），参数值来源于用户问题和每个步骤的前序步骤中的信息（如果某个参数值来自于前序工具调用的返回值，用<>标识），只需要输出有"param"字段的步骤，输出格式示例如下： [{"step": "2.1 步骤2.1", "tool": "工具名1", "param": {"参数名1": "参数值1", ...}}, {"step": "3.2 步骤3.2", "tool": "工具名2", "param": {"参数名2": "参数值2", "参数名3": "<参数值3>", ...}}, ...]\n'''
    }
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--dataset", type=str, default="plan",
        choices=["plan", 'tool_use_awareness', 'tool_selection', 'tool_selection_harder', 'tool_creation_awareness',
                 'tool_creation_awareness_harder', 'tool_creation',
                 'arguments_filling'], help="dataset used for experiment")
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )

    parser.add_argument("--english", type=str2bool, default=False)

    parser.add_argument("--self_filter", type=str2bool, default=False)

    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--model_type", type=str, default="chatglm")

    parser.add_argument("--output_dir", type=str, default="generation_test")

    parser.add_argument("--lora_path", type=str, default="")

    parser.add_argument("--iter_num", type=int, default=1)

    parser.add_argument("--sample_num", type=int, default=1)

    parser.add_argument("--cuda_ind", type=int, default=0)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=8)

    parser.add_argument("--method", type=str, default="few_shot_cot",
                        choices=["few_shot_cot", "few_shot", "zero_shot_cot", "zero_shot"])

    parser.add_argument("--load_in_8bit", type=str2bool, default=False)

    parser.add_argument("--use_typewriter", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--iter_max_new_tokens", type=int, default=1024)
    parser.add_argument("--init_max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--correct_response_format", type=str, default="The correct response is:")

    args = parser.parse_args()
    if args.english:
        if args.dataset == "plan":
            args.data_path = "eval_set_english/plan_eval.json"
            args.example_path = "eval_set_english/examples/plan_eval.json"

        elif args.dataset == "tool_use_awareness":
            args.data_path = "eval_set_english/tool_use_awareness_eval.json"
            args.example_path = "eval_set_english/examples/tool_use_awareness_eval.json"

        elif args.dataset == "tool_selection":
            args.data_path = "eval_set_english/tool_selection_eval.json"
            args.example_path = "eval_set_english/examples/tool_selection_eval.json"

        elif args.dataset == "tool_selection_harder":
            args.data_path = "eval_set_english/tool_selection_harder_eval.json"
            args.example_path = "eval_set_english/examples/tool_selection_eval.json"

        elif args.dataset == "tool_creation_awareness_harder":
            args.data_path = "eval_set_english/tool_creation_awareness_harder_eval.json"
            args.example_path = "eval_set_english/examples/tool_creation_awareness_eval.json"

        elif args.dataset == "tool_creation_awareness":
            args.data_path = "eval_set_english/tool_creation_awareness_eval.json"
            args.example_path = "eval_set_english/examples/tool_creation_awareness_eval.json"

        elif args.dataset == "tool_creation":
            args.data_path = "eval_set_english/tool_creation_eval.json"
            args.example_path = "eval_set_english/examples/tool_creation_eval.json"

        elif args.dataset == "arguments_filling":
            args.data_path = "eval_set_english/arguments_filling_eval.json"
            args.example_path = "eval_set_english/examples/arguments_filling_eval.json"

        else:
            raise ValueError("dataset is not properly defined ...")

    else:

        if args.dataset == "plan":
            args.data_path = "eval_set/plan_eval.json"
            args.example_path = "eval_set/examples/plan_eval.json"

        elif args.dataset == "tool_use_awareness":
            args.data_path = "eval_set/tool_use_awareness_eval.json"
            args.example_path = "eval_set/examples/tool_use_awareness_eval.json"

        elif args.dataset == "tool_selection":
            args.data_path = "eval_set/tool_selection_eval.json"
            args.example_path = "eval_set/examples/tool_selection_eval.json"

        elif args.dataset == "tool_selection_harder":
            args.data_path = "eval_set/tool_selection_eval_harder.json"
            args.example_path = "eval_set/examples/tool_selection_eval.json"

        elif args.dataset == "tool_creation_awareness_harder":
            args.data_path = "eval_set/tool_creation_awareness_eval_harder.json"
            args.example_path = "eval_set/examples/tool_creation_awareness_eval.json"

        elif args.dataset == "tool_creation_awareness":
            args.data_path = "eval_set/tool_creation_awareness_eval.json"
            args.example_path = "eval_set/examples/tool_creation_awareness_eval.json"

        elif args.dataset == "tool_creation":
            args.data_path = "eval_set/tool_creation_eval.json"
            args.example_path = "eval_set/examples/tool_creation_eval.json"

        elif args.dataset == "arguments_filling":
            args.data_path = "eval_set/arguments_filling_eval.json"
            args.example_path = "eval_set/examples/arguments_filling_eval.json"

        else:
            raise ValueError("dataset is not properly defined ...")

    if args.model_type == 'chatglm':
        args.model_path = '../model/chatglm3_base/chatglm3_base/'

    elif args.model_type == 'qianwen':
        args.model_path = '../model/Qwen-7B-Chat/'

    elif args.model_type == 'qianwen-13b':
        args.model_path = '../model/Qwen-14B-Chat/'

    elif args.model_type == 'qianwen-70b':
        args.model_path = '../../shijue/model/Qwen-72B-Chat/'

    elif args.model_type == 'vicuna':
        args.model_path = '../Threeagent_FastChat/model_save/vicuna_7b_v1.5/'

    elif args.model_type == 'vicuna-13b':
        args.model_path = '../model/vicuna-13b-v1.5/'

    elif args.model_type == 'llama':
        args.model_path = '../model/llama2-7b-chat/'

    elif args.model_type == 'llama-13b':
        args.model_path = '../model/Llama-2-13b-chat-hf/'

    elif args.model_type == 'llama-70b':
        args.model_path = '../../shijue/model/Llama-2-70b-chat-hf/'

    elif args.model_type == 'baichuan':
        args.model_path = '../../shijue/model/Baichuan2-7B-Chat/'

    elif args.model_type == 'baichuan-13b':
        args.model_path = '../../shijue/model/Baichuan2-13B-Chat/'

    elif args.model_type == 'mistral78':
        args.model_path = '../model/mistral-7x8-instruct/'

    else:
        raise ValueError("model type is not properly defined ...")

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."

    return args


def chinese_prompt(method, dataset, data, E=None):
    USER_PROMPT = {
        'plan': "你是一个专业的计划制定助手，给你一个用户问题，你的任务是充分理解用户问题，并制定一个合理的、可执行的多步计划以完成用户的任务，具体而言，你制订的计划应像一棵树一样分为多个子任务。输出格式为字符串（内容是使用换行符分隔的一系列子任务），例如：1. 任务1 \n 1.1 任务1.1 \n 1.2 任务1.2 \n 1.2.1 任务1.2.1 \n ... \n 2. 任务2 \n ...\n",
        'tool_use_awareness': '''你是一个专业的人工智能助手，给你一个计划，计划中有些步骤带有'tool'字段，你的任务是判断完成这些步骤是否需要使用工具，如果需要使用工具则'tool'字段为'1'，否则为'0'。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（判断，'0'或'1'）字段，例如：[{\"step\": \"1.1 步骤1.1\", \"tool\": \"0\"}, {\"step\": \"2.3 步骤2.3\", \"tool\": \"1\"}, ...]\n''',
        'tool_selection': '''你是一个专业的工具选择助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是从给定的工具集中为这些步骤选择合适的工具，并在'tool'字段填入选中工具的名称（工具的'name'字段）。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（选中工具的'name'字段）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"工具名1\"}, ...]\n''',
        'tool_selection_harder': '''你是一个专业的工具选择助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是从给定的工具集中为这些步骤选择合适的工具，并在'tool'字段填入选中工具的名称（工具的'name'字段）。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（选中工具的'name'字段）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"工具名1\"}, ...]\n''',
        'tool_creation_awareness': '''你是一个专业的人工智能助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是判断这些步骤能否在给定的工具集中找到合适的工具，如果工具集中没有合适的工具（需要创造新的工具）则'tool'字段为'1'，否则为'0'。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（判断，'0'或'1'）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 步骤4.2\", \"tool\": \"0\"}, ...]\n''',
        'tool_creation_awareness_harder': '''你是一个专业的人工智能助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是判断这些步骤能否在给定的工具集中找到合适的工具，如果工具集中没有合适的工具（需要创造新的工具）则'tool'字段为'1'，否则为'0'。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（判断，'0'或'1'）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 步骤4.2\", \"tool\": \"0\"}, ...]\n''',
        'tool_creation': '''你是一个专业的工具创造助手，给你一个计划和一个工具集，计划中有些步骤带有'tool'字段，你的任务是参照给定工具集中的工具的格式，为这些步骤创造对应的工具，并在'tool'字段填入所创造的工具。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（创造的工具，dict格式）字段，例如：[{\"step\": \"2.3 步骤2.3\", \"tool\": {\"name\": ..., \"description\": ..., \"arguments\": {\"type\": ..., \"properties\": {...}}, \"results\": {\"type\": ..., \"properties\": {...}}}}, ...]\n''',
        'arguments_filling': '''你是一个专业的工具参数填充助手，给你用户问题，问题对应的计划和相应的工具集，计划中有些步骤带有'tool'字段（工具名）和'param'字段，你的任务是先通过'tool'字段在工具集中找到对应工具，然后为这些步骤的'param'字段填入调用对应工具所需要的参数（参数格式为\"参数名=参数值\"，多个参数用','分隔），参数名来自于对应工具的'arguments'中的'properties'字段（不需要全部使用），参数值来源于用户问题和每个步骤的前序步骤中的信息（若参数值来自于前序工具调用的返回值，用<>标识）。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤），'tool'（步骤对应的'tool'字段）和'param'（填充的参数，dict格式）字段，例如： [{\"step\": \"3.2 步骤3.2\", \"tool\": \"工具名1\", \"param\": {\"参数名1\": \"参数值1\", \"参数名2\": \"<参数值2>\", ...}}, ...]\n'''
    }

    PROMPT_DICT = {
        "prompt_no_input_plan": (
            "问题：\n{input}\n"
            "输出："
        ),
        "prompt_no_input": (
            "计划：\n{input}\n"
            "输出："
        ),
        "prompt_input": (
            "计划：\n{input}\n"
            "工具集：\n{toolset}\n"
            "输出："
        ),
        "prompt_input_tool": (
            "问题：\n{question}\n"
            "计划：\n{input}\n"
            "工具集：\n{toolset}\n"
            "输出："
        ),
    }

    EXAMPLE_DICT = {
        "prompt_no_input_plan": (
            "问题：\n{input}\n"
            "输出：\n{reference}\n"
        ),
        "prompt_no_input": (
            "计划：\n{input}\n"
            "输出：\n{reference}\n"
        ),
        "prompt_input": (
            "计划：\n{input}\n"
            "工具集：\n{toolset}\n"
            "输出：\n{reference}\n"
        ),
        "prompt_input_tool": (
            "问题：\n{question}\n"
            "计划：\n{input}\n"
            "工具集：\n{toolset}\n"
            "输出：\n{reference}\n"
        ),
    }

    if method == 'zero_shot_cot':
        user_prompt = USER_PROMPT[dataset]
        if data.get("question", "") != "" and data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input_tool'].format_map(data)
        elif data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input'].format_map(data)
        else:
            user = PROMPT_DICT['prompt_no_input'].format_map(data)
        content = user_prompt + '\n' + user

    elif method == 'few_shot_cot':
        assert E != None
        user_prompt = USER_PROMPT[dataset]
        if data.get("question", "") != "" and data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input_tool'].format_map(data)
            example = EXAMPLE_DICT['prompt_input_tool'].format_map(E)
        elif data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input'].format_map(data)
            example = EXAMPLE_DICT['prompt_input'].format_map(E)
        else:
            if dataset == 'plan':
                key = 'prompt_no_input_plan'
            else:
                key = 'prompt_no_input'
            user = PROMPT_DICT[key].format_map(data)
            example = EXAMPLE_DICT[key].format_map(E)
        content = user_prompt + "\n你应该严格遵守输出格式要求，不要输出其他任何内容。\n\n样例：\n" + example + '\n让我们开始吧！\n\n' + user
    else:

        raise ('Error')
    return content


def english_prompt(method, dataset, data, E=None):
    USER_PROMPT = {
        'plan': "You are a professional planning assistant. Given a user's question, your task is to fully understand the user's question and create a reasonable, executable multi-step plan to complete the user's task. Specifically, your plan should be like a tree with multiple subtasks. The output format is a string (content is a series of subtasks separated by newline characters), for example: 1. Task 1 \n 1.1 Task 1.1 \n 1.2 Task 1.2 \n 1.2.1 Task 1.2.1 \n ... \n 2. Task 2 \n ...\n",
        'tool_use_awareness': "You are a professional AI assistant. Given a plan, some steps in the plan have a 'tool' field. Your task is to determine whether tools are needed to complete these steps. If tools are required, the 'tool' field should be '1', otherwise '0'. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (judgment, '0' or '1'), for example: [{\"step\": \"1.1 step 1.1\", \"tool\": \"0\"}, {\"step\": \"2.3 step 2.3\", \"tool\": \"1\"}, ...]\n",
        'tool_selection': "You are a professional tool selection assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to select the appropriate tool from the given toolset for these steps and fill in the 'tool' field with the name of the selected tool (the 'name' field of the tool). The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (the 'name' field of the selected tool), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"Tool Name 1\"}, ...]\n",
        'tool_selection_harder': "You are a professional tool selection assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to select the appropriate tool from the given toolset for these steps and fill in the 'tool' field with the name of the selected tool (the 'name' field of the tool). The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (the 'name' field of the selected tool), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"Tool Name 1\"}, ...]\n",
        'tool_creation_awareness': "You are a professional AI assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to determine whether it is possible to find an appropriate tool in the given toolset for these steps. If there is no suitable tool in the toolset (requiring the creation of a new tool), then the 'tool' field should be '1', otherwise '0'. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (judgment, '0' or '1'), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 step 4.2\", \"tool\": \"0\"}, ...]\n",
        'tool_creation_awareness_harder': "You are a professional AI assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to determine whether it is possible to find an appropriate tool in the given toolset for these steps. If there is no suitable tool in the toolset (requiring the creation of a new tool), then the 'tool' field should be '1', otherwise '0'. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (judgment, '0' or '1'), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 step 4.2\", \"tool\": \"0\"}, ...]\n",
        'tool_creation': "You are a professional tool creation assistant. Given a plan and a toolset, some steps in the plan have a 'tool' field. Your task is to create corresponding tools for these steps, referring to the format of tools in the given toolset, and fill in the 'tool' field with the created tool. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (the created tool, dict format), for example: [{\"step\": \"2.3 step 2.3\", \"tool\": {\"name\": ..., \"description\": ..., \"arguments\": {\"type\": ..., \"properties\": {...}}, \"results\": {\"type\": ..., \"properties\": {...}}}}, ...]\n",
        'arguments_filling': "You are a professional tool parameter filling assistant. Given a user's question, the corresponding plan, and a set of tools, some steps in the plan have a 'tool' field (tool name) and a 'param' field. Your task is to first find the corresponding tool in the toolset through the 'tool' field, and then fill in the 'param' field for these steps with the parameters required to call the corresponding tool (parameter format is \"parameter name=parameter value\", separate multiple parameters with ','), where the parameter names come from the 'properties' field in the 'arguments' of the corresponding tool (not all need to be used), and parameter values come from the user's question and information from the previous steps (if the parameter value comes from the return value of a previous tool call, use <> to indicate). The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field), 'tool' (the corresponding 'tool' field of the step), and 'param' (the filled parameters, dict format), for example: [{\"step\": \"3.2 step 3.2\", \"tool\": \"Tool Name 1\", \"param\": {\"parameter name1\": \"parameter value1\", \"parameter name2\": \"<parameter value2>\", ...}}, ...]\n"
    }

    PROMPT_DICT = {
        "prompt_no_input_plan": (
            "Question: \n{input}\n"
            "Output: "
        ),
        "prompt_no_input": (
            "Plan: \n{input}\n"
            "Output: "
        ),
        "prompt_input": (
            "Plan: \n{input}\n"
            "Toolset: \n{toolset}\n"
            "Output: "
        ),
        "prompt_input_tool": (
            "Question: \n{question}\n"
            "Plan: \n{input}\n"
            "Toolset: \n{toolset}\n"
            "Output: "
        ),
    }

    EXAMPLE_DICT = {
        "prompt_no_input_plan": (
            "Question: \n{input}\n"
            "Output: \n{reference}\n"
        ),
        "prompt_no_input": (
            "Plan: \n{input}\n"
            "Output: \n{reference}\n"
        ),
        "prompt_input": (
            "Plan: \n{input}\n"
            "Toolset: \n{toolset}\n"
            "Output: \n{reference}\n"
        ),
        "prompt_input_tool": (
            "Question: \n{question}\n"
            "Plan: \n{input}\n"
            "Toolset: \n{toolset}\n"
            "Output: \n{reference}\n"
        ),
    }

    if method == 'zero_shot_cot':
        user_prompt = USER_PROMPT[dataset]
        if data.get("question", "") != "" and data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input_tool'].format_map(data)
        elif data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input'].format_map(data)
        else:
            user = PROMPT_DICT['prompt_no_input'].format_map(data)
        content = user_prompt + '\n' + user

    elif method == 'few_shot_cot':
        assert E != None
        user_prompt = USER_PROMPT[dataset]
        if data.get("question", "") != "" and data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input_tool'].format_map(data)
            example = EXAMPLE_DICT['prompt_input_tool'].format_map(E)
        elif data.get("toolset", "") != "":
            user = PROMPT_DICT['prompt_input'].format_map(data)
            example = EXAMPLE_DICT['prompt_input'].format_map(E)
        else:
            if dataset == 'plan':
                key = 'prompt_no_input_plan'
            else:
                key = 'prompt_no_input'
            user = PROMPT_DICT[key].format_map(data)
            example = EXAMPLE_DICT[key].format_map(E)
        content = user_prompt + "\nYou should strictly follow the output format requirements and not output any other content. \n\n Example: \n" + example + '\nLet’s Begin!\n\n' + user
    else:
        raise ('Error')
    return content


def generate_prompt_ultratool(args, data, E=None):
    if not args.english:
        content = chinese_prompt(args.method, args.dataset, data, E)

    else:
        content = english_prompt(args.method, args.dataset, data, E)

    MODEL_DICT = {
        "llama": (
            "[INST] \n{content}\n [/INST]"
        ),
        "mistral": (
            "<s>[INST] {content} [/INST]"
        ),
        "chatglm": (
            "<|user|> \n{content}\n <|assistant|>"
        ),
        "qianwen": (
            "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "baichuan": (
            "<reserved_106>{content}<reserved_107>"
        )

    }

    if args.model_type in ["qianwen", "qianwen-13b", "qianwen-70b"]:
        message = MODEL_DICT['qianwen'].format_map(
            {'content': content}
        )
        return message

    elif args.model_type in ["chatglm"]:

        return content

    elif args.model_type in ["llama", "llama-13b", 'llama-70b']:
        message = MODEL_DICT['llama'].format_map(
            {'content': content}
        )

        return message

    elif "mistral" in args.model_type:
        message = MODEL_DICT['mistral'].format_map(
            {'content': content}
        )

        return message


    elif args.model_type in ["vicuna", "original_vicuna", "vicuna-13b"]:
        conv.messages = []
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    elif "baichuan" in args.model_type:
        message = MODEL_DICT['baichuan'].format_map(
            {'content': content}
        )

        return message



    else:
        raise ValueError("we do not implement prompt for such model type yet")


from collections import Counter


def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            data.append(line)
    return data


def get_question_answer(args):
    questions = read_data(args.data_path)
    answers = []

    for item in questions:
        try:
            answers.append(item['reference'])
        except:
            item['reference'] = item['referencce']
            answers.append(item['reference'])

    return questions, answers


def main2(args):
    from vllm import LLM, SamplingParams
    import torch

    model = LLM(model=args.model_path, dtype="float16", trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel)
    print(args.model_path)

    if 'qianwen' in args.model_type:
        model.llm_engine.tokenizer.eos_token_id = 151645
        model.llm_engine.tokenizer.pad_token_id = None

    print("load data")

    questions, answers = get_question_answer(args)

    qa_pairs = [(questions[idx], answers[idx]) for idx in range(len(questions))]
    example = json.load(open(args.example_path, 'r', encoding='utf-8'))

    cuda_pieces = np.array_split(range(len(qa_pairs)), args.cuda_num // args.tensor_parallel)

    if not os.path.exists(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            pass

    total_output = {}

    with open(f"{args.output_dir}/{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json", "w",
              encoding='utf-8') as wf:
        start = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][0]
        end = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][-1] + 1
        total_line = 0
        for (question, answer) in tqdm(qa_pairs[start:end]):

            prompt = generate_prompt_ultratool(args, question, example)

            with torch.no_grad():
                output_list = []
                try:
                    for i in range(args.sample_num):
                        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                                         max_tokens=args.init_max_new_tokens)
                        generation_output = model.generate(prompt, sampling_params, use_tqdm=False)
                        output = generation_output[0].outputs[0].text
                        output_list.append(output)
                        del generation_output

                except Exception as e:
                    output_list.append(str(e))

                total_output['init prompt'] = prompt
                total_output['init output'] = output_list

                dict = {
                    "question": question,
                    "original ans": answer,
                    "total output": total_output,
                }
                wf.writelines(json.dumps(dict, ensure_ascii=False) + '\n')
                total_line += 1
                if total_line % 5 == 0:
                    wf.flush()


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))


def main(argv=None):
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    main2(args)


if __name__ == "__main__":
    main()
