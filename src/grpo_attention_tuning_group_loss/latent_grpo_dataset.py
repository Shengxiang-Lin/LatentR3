import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import csv
import ast

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def get_prefix_data(info_file, tokenizer):
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        # 千万别把下面这两行对齐，就这个写法，不然会多出空格
        # Don't align the next two lines, otherwise it will add an extra space!
        info = [f'''### Response: 
{_}''' for _ in info]
    prefixID = [tokenizer(_).input_ids for _ in info]
    thought_id = len(tokenizer) - 1
    for i in range(len(prefixID)):
        prefixID[i].insert(4, thought_id)
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(5, len(ID)):
            if i == 5:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[5:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    return hash_dict

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)
    
def count_product_occurrences(csv_file, txt_file):
    '''
    Count the number of times a product appears in the CSV file history_item_title.
    
    Parameters:
        csv_file: CSV file path.
        txt_file: TXT file path containing product names and numbers.
    Returns:
        dict: Dictionary with product names as keys and their occurrence counts as values.
    '''
    product_counts = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                product_name = parts[0].strip()
                product_counts[product_name] = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history_titles = ast.literal_eval(row['history_item_title'])
            for title in history_titles:
                cleaned_title = title.strip()
                if cleaned_title in product_counts:
                    product_counts[cleaned_title] += 1
    return product_counts

class D3Dataset(Dataset):
    def __init__(self, train_file, tokenizer, info_file, max_len=2048, sample=-1, test=False, seed=0, category="", K=4,
                 dedup=False, num_groups=3):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.K = K
        self.dedup = dedup
        self.num_groups = num_groups
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        ]
        # 新增：计算产品出现次数
        self.product_counts = count_product_occurrences(train_file, info_file)   
        # 新增：预计算所有样本的total_count并确定分组阈值
        total_counts = []
        for idx in range(len(self.data)):
            history_titles = ast.literal_eval(self.data.iloc[idx]['history_item_title'])
            total_count = sum(self.product_counts.get(title.strip(), 0) for title in history_titles)
            total_counts.append(total_count)
        self.total_counts = np.array(total_counts)
        # 计算分位数阈值（例如，num_groups=3时，取1/3和2/3分位数）
        if len(total_counts) > 0:
            quantiles = np.quantile(total_counts, np.linspace(0, 1, num_groups + 1)[1:-1])
            self.group_thresholds = quantiles.tolist()
        else:
            self.group_thresholds = []
        self.get_inputs()

    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title'])
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ", \"" + row['history_item_title'][i] + "\""
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}

    def pre(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[0]}
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        thought_idx = len(tokens)
        tokens = tokens + self.tokenizer.encode("<|Thought|>",bos=False,eos=False)
        attention_mask = len(tokens) * [1]
        # 新增：计算总出现次数并分配组
        history_titles = ast.literal_eval(self.data.iloc[idx]['history_item_title'])
        total_count = sum(self.product_counts.get(title.strip(), 0) for title in history_titles)
        # 根据total_count分配组
        if not self.group_thresholds:  # 空数据集或无阈值
            group = 0
        else:
            group = 0
            for i, threshold in enumerate(self.group_thresholds):
                if total_count > threshold:
                    group = i + 1
                else:
                    break
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "target": target_item,
                "group": group,
            }
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        if len(tokens) >= self.max_len:
            print(len(tokens))
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "group": group, 
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]