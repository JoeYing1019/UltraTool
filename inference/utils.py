import json

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


def generate_prompt_chinese(method, dataset, data, E=None):
    USER_PROMPT = {
        'planning': "你是一个专业的计划制定助手，给你一个用户问题，你的任务是充分理解用户问题，并制定一个合理的、可执行的多步计划以完成用户的任务，具体而言，你制订的计划应像一棵树一样分为多个子任务。输出格式为字符串（内容是使用换行符分隔的一系列子任务），例如：1. 任务1 \n 1.1 任务1.1 \n 1.2 任务1.2 \n 1.2.1 任务1.2.1 \n ... \n 2. 任务2 \n ...\n",
        'tool_usage_awareness': '''你是一个专业的人工智能助手，给你一个计划，计划中有些步骤带有'tool'字段，你的任务是判断完成这些步骤是否需要使用工具，如果需要使用工具则'tool'字段为'1'，否则为'0'。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（判断，'0'或'1'）字段，例如：[{\"step\": \"1.1 步骤1.1\", \"tool\": \"0\"}, {\"step\": \"2.3 步骤2.3\", \"tool\": \"1\"}, ...]\n''',
        'tool_selection': '''你是一个专业的工具选择助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是从给定的工具集中为这些步骤选择合适的工具，并在'tool'字段填入选中工具的名称（工具的'name'字段）。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（选中工具的'name'字段）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"工具名1\"}, ...]\n''',
        'tool_creation_awareness': '''你是一个专业的人工智能助手，给你一个计划和相应的工具集，计划中有些步骤带有'tool'字段，你的任务是判断这些步骤能否在给定的工具集中找到合适的工具，如果工具集中没有合适的工具（需要创造新的工具）则'tool'字段为'1'，否则为'0'。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（判断，'0'或'1'）字段，例如：[{\"step\": \"2.1 步骤2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 步骤4.2\", \"tool\": \"0\"}, ...]\n''',
        'tool_creation': '''你是一个专业的工具创造助手，给你一个计划和一个工具集，计划中有些步骤带有'tool'字段，你的任务是参照给定工具集中的工具的格式，为这些步骤创造对应的工具，并在'tool'字段填入所创造的工具。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤）和'tool'（创造的工具，dict格式）字段，例如：[{\"step\": \"2.3 步骤2.3\", \"tool\": {\"name\": ..., \"description\": ..., \"arguments\": {\"type\": ..., \"properties\": {...}}, \"results\": {\"type\": ..., \"properties\": {...}}}}, ...]\n''',
        'tool_usage': '''你是一个专业的工具参数填充助手，给你用户问题，问题对应的计划和相应的工具集，计划中有些步骤带有'tool'字段（工具名）和'param'字段，你的任务是先通过'tool'字段在工具集中找到对应工具，然后为这些步骤的'param'字段填入调用对应工具所需要的参数（参数格式为\"参数名=参数值\"，多个参数用','分隔），参数名来自于对应工具的'arguments'中的'properties'字段（不需要全部使用），参数值来源于用户问题和每个步骤的前序步骤中的信息（若参数值来自于前序工具调用的返回值，用<>标识）。输出格式为元素是dict的list，每个dict包含'step'（给定计划中有'tool'字段的步骤），'tool'（步骤对应的'tool'字段）和'param'（填充的参数，dict格式）字段，例如： [{\"step\": \"3.2 步骤3.2\", \"tool\": \"工具名1\", \"param\": {\"参数名1\": \"参数值1\", \"参数名2\": \"<参数值2>\", ...}}, ...]\n'''
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
        raise ('error')
    return content

def generate_prompt_english(method, dataset, data, E=None):
    USER_PROMPT = {
        'planning': "You are a professional planning assistant. Given a user's question, your task is to fully understand the user's question and create a reasonable, executable multi-step plan to complete the user's task. Specifically, your plan should be like a tree with multiple subtasks. The output format is a string (content is a series of subtasks separated by newline characters), for example: 1. Task 1 \n 1.1 Task 1.1 \n 1.2 Task 1.2 \n 1.2.1 Task 1.2.1 \n ... \n 2. Task 2 \n ...\n",
        'tool_usage_awareness': "You are a professional AI assistant. Given a plan, some steps in the plan have a 'tool' field. Your task is to determine whether tools are needed to complete these steps. If tools are required, the 'tool' field should be '1', otherwise '0'. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (judgment, '0' or '1'), for example: [{\"step\": \"1.1 step 1.1\", \"tool\": \"0\"}, {\"step\": \"2.3 step 2.3\", \"tool\": \"1\"}, ...]\n",
        'tool_selection': "You are a professional tool selection assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to select the appropriate tool from the given toolset for these steps and fill in the 'tool' field with the name of the selected tool (the 'name' field of the tool). The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (the 'name' field of the selected tool), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"Tool Name 1\"}, ...]\n",
        'tool_creation_awareness': "You are a professional AI assistant. Given a plan and a corresponding set of tools, some steps in the plan have a 'tool' field. Your task is to determine whether it is possible to find an appropriate tool in the given toolset for these steps. If there is no suitable tool in the toolset (requiring the creation of a new tool), then the 'tool' field should be '1', otherwise '0'. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (judgment, '0' or '1'), for example: [{\"step\": \"2.1 step 2.1\", \"tool\": \"1\"}, {\"step\": \"4.2 step 4.2\", \"tool\": \"0\"}, ...]\n",
        'tool_creation': "You are a professional tool creation assistant. Given a plan and a toolset, some steps in the plan have a 'tool' field. Your task is to create corresponding tools for these steps, referring to the format of tools in the given toolset, and fill in the 'tool' field with the created tool. The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field) and 'tool' (the created tool, dict format), for example: [{\"step\": \"2.3 step 2.3\", \"tool\": {\"name\": ..., \"description\": ..., \"arguments\": {\"type\": ..., \"properties\": {...}}, \"results\": {\"type\": ..., \"properties\": {...}}}}, ...]\n",
        'tool_usage': "You are a professional tool parameter filling assistant. Given a user's question, the corresponding plan, and a set of tools, some steps in the plan have a 'tool' field (tool name) and a 'param' field. Your task is to first find the corresponding tool in the toolset through the 'tool' field, and then fill in the 'param' field for these steps with the parameters required to call the corresponding tool (parameter format is \"parameter name=parameter value\", separate multiple parameters with ','), where the parameter names come from the 'properties' field in the 'arguments' of the corresponding tool (not all need to be used), and parameter values come from the user's question and information from the previous steps (if the parameter value comes from the return value of a previous tool call, use <> to indicate). The output format is a list of dicts, each dict contains 'step' (the step in the given plan that has a 'tool' field), 'tool' (the corresponding 'tool' field of the step), and 'param' (the filled parameters, dict format), for example: [{\"step\": \"3.2 step 3.2\", \"tool\": \"Tool Name 1\", \"param\": {\"parameter name1\": \"parameter value1\", \"parameter name2\": \"<parameter value2>\", ...}}, ...]\n"
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
        raise ('error')
    return content