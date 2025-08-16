import json
import csv
import ast
import math
from collections import defaultdict

def count_product_occurrences(csv_file, txt_file):
    '''
    Count the number of times a product appears in the CSV file history_item_title.
    
    Parameters:
        csv_file: CSV file path.
        txt_file: TXT file path containing product names and numbers.
    Returns:
        dict: Dictionary with product names as keys and their occurrence counts as values.
    '''
    product_counts = defaultdict(int)
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                product_name = parts[0].strip()
                product_counts[product_name] = 0  # 初始化为0
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history_titles = ast.literal_eval(row['history_item_title'])
            for title in history_titles:
                cleaned_title = title.strip()
                if cleaned_title in product_counts:
                    product_counts[cleaned_title] += 1
    return product_counts

# 主程序
if __name__ == "__main__":
    csv_file = 'data/test/CDs_and_Vinyl_5_2015-10-2018-11.csv'
    txt_file = 'data/info/CDs_and_Vinyl_5_2015-10-2018-11.txt'  # 假设info文件路径
    json_file = 'output_dir/CDs_and_Vinyl/nomul_attn_group_lr_2_5e-5/final_result.json'
    n = 3  # 分组数量，可以根据需要调整

    # 计算每个物品的出现次数
    product_counts = count_product_occurrences(csv_file, txt_file)

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    ks = [1, 3, 5, 10]  # K值列表

    for item in data:
        # 解析input中的历史物品列表
        input_str = item['input']
        prefix = "The user has palyed the following musicss before: "
        if input_str.startswith(prefix):
            history_str = input_str[len(prefix):]
            history = [t.strip().strip('"') for t in history_str.split(', ')]
        else:
            history = []

        # 计算活跃度：历史物品出现次数之和
        activity_sum = sum(product_counts.get(t, 0) for t in history)

        # 清洗ground truth
        gt = item['output'].strip().strip('"').strip('\n').strip()

        # 清洗predict列表
        predicts = [p.strip().strip('"').strip('\n').strip() for p in item['predict']]

        # 计算每个K的HR和NDCG
        metrics = {}
        found_rank = None
        for rank, pred in enumerate(predicts, 1):
            if pred == gt:
                found_rank = rank
                break

        for k in ks:
            hr_k = 1 if found_rank is not None and found_rank <= k else 0
            ndcg_k = 1 / math.log2(found_rank + 1) if found_rank is not None and found_rank <= k else 0
            metrics[f'hr@{k}'] = hr_k
            metrics[f'ndcg@{k}'] = ndcg_k

        records.append({
            'activity_sum': activity_sum,
            'metrics': metrics
        })

    # 按活跃度降序排序（活跃度越高越热门）
    records.sort(key=lambda x: x['activity_sum'], reverse=True)

    # 分成n组，每组尽量均匀
    total_records = len(records)
    group_size = total_records // n
    groups = []
    for i in range(n):
        start = i * group_size
        end = start + group_size if i < n - 1 else total_records
        group_records = records[start:end]
        
        if group_records:
            avg_metrics = {key: sum(r['metrics'][key] for r in group_records) / len(group_records) for key in group_records[0]['metrics']}
        else:
            avg_metrics = {f'hr@{k}': 0 for k in ks}
            avg_metrics.update({f'ndcg@{k}': 0 for k in ks})
        
        groups.append({
            'group': i + 1,  # 组号1最热门
            'avg_metrics': avg_metrics
        })

    # 输出每个组的指标
    for g in groups:
        print(f"Group {g['group']} (most popular to least):")
        for k in ks:
            print(f"  HR@{k}: {g['avg_metrics'][f'hr@{k}']:.4f}")
            print(f"  NG@{k}: {g['avg_metrics'][f'ndcg@{k}']:.4f}")