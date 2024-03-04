import datetime
import json
import os
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from tqdm import tqdm
import openai
import tiktoken

API_list = [
    "",
]
API_name_key_list = [
    "key1",
]

class Timer(object):
    def __init__(self):
        self.__start = time.time()

    def start(self):
        self.__start = time.time()

    def get_time(self, restart=True, format=False):
        end = time.time()
        span = end - self.__start
        if restart:
            self.__start = end
        if format:
            return self.format(span)
        else:
            return span

    def format(self, seconds):
        return datetime.timedelta(seconds=int(seconds))

    def print(self, name):
        print(name, self.get_time())


API_ID = 0
lock = threading.Lock()


def set_next_API_ID():
    global API_ID
    lock.acquire()
    API_ID = (API_ID + 1) % len(API_name_key_list)
    openai.api_key = API_list[API_ID]
    lock.release()


def write_result(result, outf):
    lock.acquire()
    outf.write(json.dumps(result, ensure_ascii=False) + '\n')
    outf.flush()
    lock.release()


def multi_threading_running(func, queries, outf, engine='gpt-4-1106-preview', n=50, multiple_API=True):
    def wrapped_function(line, max_try=50):
        if multiple_API:
            set_next_API_ID()
        try:
            query = line['prompt_new']
            result = func(query,engine)
            if 'choices' in result and result['choices'][0]['message']['content']:
                line.update({'model_response':result['choices'][0]['message']['content']})
                write_result(line,outf)
            return line
        except Exception as e:
            print(f'retried_times: {20-max_try}, API_key: {openai.api_key}\n Error: '+str(e))
            if not isinstance(e, openai.error.RateLimitError):
                if isinstance(e, openai.error.APIError):
                    print("API Error")
                else:
                    print("found a error:", e)
            else:
                time.sleep(5)
            if max_try > 0:
                return wrapped_function(line, max_try-1)

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(tqdm(executor.map(wrapped_function, queries), total=len(queries)))
    return results
   

cache = {}

def query_azure_openai_chat(query, engine="gpt-4-1106-preview"):
    global cache
    query_string = json.dumps(query) if not isinstance(query, str) else query
    if query_string in cache:
        return cache[query_string]
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
    ]
    if isinstance(query, str):
        messages.append(
            {"role": "user", "content": query},
        )
    elif isinstance(query, list):
        messages += query
    else:
        raise ValueError("Unsupported query: {0}".format(query))
    if '16k' in engine:
        max_tokens = 10000
    else:
        max_tokens = 4096
    response = openai.ChatCompletion.create(
        model=engine,  
        messages=messages,
        temperature=0,
        max_tokens=max_tokens
    )

    if response is None:
        raise Exception(f"Sorry, there is no response!")

    if response['choices'][0]["finish_reason"] != "stop":
        raise Exception(f"Sorry, response has error due to {response['choices'][0]['finish_reason']}")

    try:
        if response['choices'][0]['message']['content'] != "":
            cache[query_string] = response
    except Exception as e:
        pass

    return response


def multi_threading_gpt4_infer(data, args, outf, model_name="gpt-4-1106-preview", max_workers=50):
    timer = Timer()
    print('Processing queires')
    results = multi_threading_running(query_azure_openai_chat, data, outf, engine=model_name, n=max_workers, multiple_API=True)
    print("Average time after {0} samples: {1}".format(len(data), timer.get_time(restart=False) / len(data)))
    print('Processed queries')
    return results
