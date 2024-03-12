import argparse
from tqdm import tqdm
import json
from utils import generate_prompt_chinese, generate_prompt_english, read_data
from conversation import Conversation, SeparatorStyle
from collections import Counter
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--dataset", type=str, default="",
        choices=["planning", 'tool_usage_awareness', 'tool_selection', 'tool_creation_awareness', 'tool_creation', 'tool_usage'], help="dataset used for experiment")
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought")
    parser.add_argument("--english", type=str2bool, default=False)
    parser.add_argument("--self_filter", type=str2bool, default=False)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", type=str, default="chatglm")
    parser.add_argument("--output_dir", type=str, default="predictions")
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
        test_path = 'data/English-dataset/test_set'
        example_path = 'data/English-dataset/example'
        args.data_path = os.path.join(test_path, '{}.json'.format(args.dataset))
        args.example_path = os.path.join(example_path, '{}.json'.format(args.dataset))

    else:
        test_path = 'data/Chinese-dataset/test_set'
        example_path = 'data/English-dataset/example'
        args.data_path = os.path.join(test_path, '{}.json'.format(args.dataset))
        args.example_path = os.path.join(example_path, '{}.json'.format(args.dataset))

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."

    return args


def generate_prompt_ultratool(args, data, E=None):
    if not args.english:
        content = generate_prompt_chinese(args.method, args.dataset, data, E)

    else:
        content = generate_prompt_english(args.method, args.dataset, data, E)

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
