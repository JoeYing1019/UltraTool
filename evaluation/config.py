
ch_prompt_plan = '''
作为一名专业评估专家，你的任务是参照标准答案，根据给定的评估维度客观评估所提供数据的质量。给定用户指令，标准答案和用户指令对应的任务规划，请根据以下评估维度对任务规划的质量进行评分：

1. 准确性：任务规划与用户指令的目标应该一致，对用户指令的理解和用户指令中提供信息的使用需要准确，不能虚增用户指令中没有要求的不合理任务或不合理约束。
2. 完整性：用户指令中涉及的多个任务、约束全部需要在任务规划的步骤中体现，不能有遗漏。
3. 可执行性：任务规划整体逻辑连贯，任务规划中的步骤均合理且可执行，步骤之间顺序合理，可以逐步执行以解决用户指令，不会缺失步骤导致后续步骤无法执行，也不会有冗余步骤导致执行错误。
4. 语法健全性：任务规划的内容语法健全，语句通顺流畅，语言风格良好，没有语法错误。
5. 结构合理性：任务规划的结构是一个有序的树状结构，父操作和子操作之间关系合理，整体组织结构高效合理。
6. 高效性：任务规划简洁高效，步骤清晰明确，不会过度细分步骤，没有冗长复杂的流程。

总的来说，模型回答的质量越高，则分数越高，作为示例，标准答案在各个维度和总分上都可以得到8分。
对照标准答案，逐步给上述每个评估维度打一个分数，然后根据所有的评估维度，打一个总分，总分的具体打分标准为：
当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1分；
当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为2到3分；
当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得4到6分；
当模型回答质量与标准答案相近，在所有维度上表现良好，总分得7到8分；
只有当模型回答质量显著超过标准答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。

你必须按照以下格式提供你的评估结果：
[
    {{"打分理由": <对照准确性定义和标准答案，提供评分原因>, "准确性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照完整性定义和标准答案，提供评分原因>, "完整性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照可执行性定义和标准答案，提供评分原因>, "可执行性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照语法健全性定义和标准答案，提供评分原因>, "语法健全性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照结构合理性定义和标准答案，提供评分原因>, "结构合理性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照高效性定义和标准答案，提供评分原因>, "高效性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照所有评估维度定义和标准答案，提供评分原因>, "总分": <分配1到10分之间的分数>}},
]

以下是给定的用户指令，标准答案和要评估的任务规划：
用户指令：{question}
标准答案：{reference}
要评估的任务规划：{answer}

根据上述评估维度，对照标准答案，对要评估的任务规划逐步给每个评估维度打分，然后再打一个总分。最终输出内容应以json字符串的形式输出，不要输出其他任何内容。
输出：
'''

en_prompt_plan = '''
As a professional assessment expert, your task is to objectively evaluate the quality of the provided data based on the given assessment dimensions, with reference to the standard answer. Given user instructions, a standard answer, and a task planning corresponding to the user instructions, please score the quality of the task planning according to the following assessment dimensions:

1. Accuracy: The task planning should align with the objectives of the user instructions. The understanding of the user instructions and the use of information provided within them must be accurate, without adding unreasonable tasks or constraints that are not requested by the user.
2. Completeness: All tasks and constraints involved in the user instructions must be reflected in the steps of task planning without omissions.
3. Executability: The overall logic of the task planning should be coherent. All steps in the task planning should be reasonable and executable, with a logical sequence that allows for gradual completion to address the user's instructions. There should be no missing steps that would prevent subsequent steps from being executed, nor any superfluous steps that could cause errors in execution.
4. Syntactic Soundness: The content of the task planning should be grammatically sound, with smooth and fluent sentences, a good language style, and free of grammatical errors.
5. Structural Rationality: The structure of the task planning should be an ordered tree-like hierarchy, with reasonable relationships between parent and child operations, and an overall efficient and rational organization.
6. Efficiency: The task planning should be concise and efficient, with clear and specific steps, without excessively subdividing steps or having lengthy and complicated procedures.

Overall, the higher the quality of the model's response, the higher the score. As an example, the standard answer could receive a score of 8 in each dimension and in total.
Contrasting with the standard answer, assign a score to each of the above assessment dimensions individually, and then give an overall score based on all the assessment dimensions. The specific criteria for the overall score are as follows:
The overall score must be 1 if the model's response is irrelevant to the question, contains fundamental factual errors, or generates harmful content.
If the model's response has no serious errors and is generally harmless but of low quality and does not meet the user's needs, the overall score should be between 2 and 3.
If the model's response basically meets the user's requirements but performs poorly on some dimensions, with medium quality, the overall score should be between 4 and 6.
If the model's response is close to the quality of the standard answer and performs well in all dimensions, the overall score should be between 7 and 8.
Only if the model's response significantly surpasses the standard answer, thoroughly addresses the user's questions and all needs, and is near perfect in all dimensions, can it receive a score between 9 and 10.

You must provide your assessment results in the following format:
[
{{"Reasoning": <Provide reasoning for the score with reference to the accuracy definition and standard answer>, "Accuracy Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to the completeness definition and standard answer>, "Completeness Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to the executability definition and standard answer>, "Executability Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to the syntactic soundness definition and standard answer>, "Syntactic Soundness Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to the structural rationality definition and standard answer>, "Structural Rationality Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to the efficiency definition and standard answer>, "Efficiency Score": <Assign a score between 1 and 10>}},
{{"Reasoning": <Provide reasoning for the score with reference to all assessment dimension definitions and standard answer>, "Overall Score": <Assign a score between 1 and 10>}},
]

Here are the given user instructions, standard answer, and the task planning to be assessed:
User Instructions: {question}
Standard Answer: {reference}
Task Planning to be Assessed: {answer}

Based on the above assessment dimensions and contrasting with the standard answer, score each assessment dimension for the task planning to be assessed, and then give an overall score. The final output should be in the form of a JSON string, without including any other content.
Output:
'''

ch_prompt_tool = '''
作为一名专业评估专家，你的任务是参照标准答案，根据给定的评估维度客观评估所提供数据的质量。给定用户指令，标准答案和根据用户指令所创造的工具，请根据以下评估维度对创造的工具的质量进行评分：

1. 格式遵从性：创造的工具应该与标准答案在格式上保持完全一致，完整包含工具名（"name"字段）、工具描述（"description"字段）、参数列表（"arguments"字段，"arguments"中还包含"type"和"properties"字段）和返回值（"results"字段，"results"中还包含"type"和"properties"字段）这几个基本组成部分。
2. 准确性：创造的工具与用户指令的目标应该一致，能够准确解决用户指令的需求。
3. 内容合理性：创造的工具定义的各个字段的内容都应该合理，包括自然语言描述字段的表达清晰、语法健全，以及定义的各个参数的类型和各个返回值的类型均是合理的。
4. 可执行性：创造的工具定义的工具名和工具描述能够恰当表达工具的功能，定义的参数列表完备，定义的返回值完备。
5. 丰富度：创造的工具包含丰富的信息、深度、上下文考虑和多样性。


总的来说，模型回答的质量越高，则分数越高，作为示例，标准答案在各个维度和总分上都可以得到8分。
对照标准答案，逐步给上述每个评估维度打一个分数，然后根据所有的评估维度，打一个总分，总分的具体打分标准为：
当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1分；
当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为2到3分；
当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得4到6分；
当模型回答质量与标准答案相近，在所有维度上表现良好，总分得7到8分；
只有当模型回答质量显著超过标准答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。

你必须按照以下格式提供你的评估结果：
[
    {{"打分理由": <对照格式遵从性定义和标准答案，提供评分原因>, "格式遵从性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照准确性定义和标准答案，提供评分原因>, "准确性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照内容合理性定义和标准答案，提供评分原因>, "内容合理性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照可执行性定义和标准答案，提供评分原因>, "可执行性分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照丰富度定义和标准答案，提供评分原因>, "丰富度分数": <分配1到10分之间的分数>}},
    {{"打分理由": <对照所有评估维度定义和标准答案，提供评分原因>, "总分": <分配1到10分之间的分数>}},
]

以下是给定的用户指令，标准答案和要评估的创造的工具：
用户指令：{question}
标准答案：{reference}
创造的工具：{answer}

根据上述评估维度，对照标准答案，对要评估的创造的工具逐步给每个评估维度打分，然后再打一个总分。最终输出内容应以json字符串的形式输出，不要输出其他任何内容。
输出：
'''

en_prompt_tool = '''
As a professional assessment expert, your task is to objectively assess the quality of the provided data in reference to the standard answer, based on the given assessment dimensions. Given a user instruction, the standard answer, and a tool created in response to the user instruction, please score the quality of the created tool according to the following assessment dimensions:

1. Format Compliance: The created tool should be completely consistent with the standard answer in terms of format, fully including the basic components such as the tool name ("name" field), tool description ("description" field), list of arguments ("arguments" field, with "type" and "properties" fields within "arguments") and return values ("results" field, with "type" and "properties" fields within "results").
2. Accuracy: The created tool should align with the objectives of the user instruction and accurately address the user's needs.
3. Content Reasonableness: The content within each field of the created tool should be reasonable, including clear expression and solid grammar in the natural language description fields, as well as sensible types for each defined argument and return value.
4. Executability: The tool name and description defined in the created tool should appropriately express its function, with a comprehensive list of parameters and complete return values.
5. Richness: The created tool should include rich information, depth, contextual considerations, and diversity.

Overall, the higher the quality of the model answer, the higher the score. As an example, the standard answer can score 8 points in each dimension and in total.
Compare the standard answer and step by step score each of the above assessment dimensions, then provide an overall score based on all dimensions. The specific criteria for the overall score are as follows:
The total score must be 1 point if the model answer is irrelevant to the question, contains essential factual errors, or generates harmful content.
The total score should be 2 to 3 points if the model answer is of low quality without serious errors and is harmless but does not meet user needs.
The total score can be 4 to 6 points if the model answer generally meets user requirements but performs poorly in some dimensions and is of mediocre quality.
The total score should be 7 to 8 points if the model answer's quality is close to the standard answer and performs well in all dimensions.
A score of 9 to 10 points is only achievable if the model answer significantly surpasses the standard answer, fully resolves the user's issue and all requirements, and approaches a perfect score in all dimensions.

You must provide your assessment results in the following format:
[
{{"Scoring Reason": <Provide reasons for scoring against the definition of format compliance and the standard answer>, "Format Compliance Score": <Assign a score between 1 to 10>}},
{{"Scoring Reason": <Provide reasons for scoring against the definition of accuracy and the standard answer>, "Accuracy Score": <Assign a score between 1 to 10>}},
{{"Scoring Reason": <Provide reasons for scoring against the definition of content reasonableness and the standard answer>, "Content Reasonableness Score": <Assign a score between 1 to 10>}},
{{"Scoring Reason": <Provide reasons for scoring against the definition of executability and the standard answer>, "Executability Score": <Assign a score between 1 to 10>}},
{{"Scoring Reason": <Provide reasons for scoring against the definition of richness and the standard answer>, "Richness Score": <Assign a score between 1 to 10>}},
{{"Scoring Reason": <Provide reasons for scoring against all assessment dimensions and the standard answer>, "Total Score": <Assign a score between 1 to 10>}},
]

Below are the given user instruction, standard answer, and the created tool to be evaluated:
User instruction: {question}
Standard answer: {reference}
Created tool: {answer}

Based on the above assessment dimensions and comparing against the standard answer, score each dimension for the created tool to be evaluated, then provide an overall score. The final output should be in the form of a JSON string, without any additional content.
Output:
'''



language_to_path = {
    "ch" : "predictions/Chinese-dataset",
    "en" : "predictions/English-dataset"
}

language_to_plan_eval_prompt = {
    "ch" : ch_prompt_plan,
    "en" : en_prompt_plan
}

language_to_tool_eval_prompt = {
    "ch" : ch_prompt_tool,
    "en" : en_prompt_tool
}