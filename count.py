import csv
import ast

def count_product_occurrences(csv_file, txt_file, output_file=None):
    '''
    Count the number of times a product appears in the CSV file history_item_title.
    
    Parameters:
        csv_file: CSV file path.
        txt_file: TXT file path containing product names and numbers.
        output_file: Result output file path (optional).
    '''
    product_counts = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                product_name = parts[0]
                product_counts[product_name] = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history_titles = ast.literal_eval(row['history_item_title'])
            for title in history_titles:
                cleaned_title = title.strip()
                if cleaned_title in product_counts:
                    product_counts[cleaned_title] += 1
    results = []
    for product, count in product_counts.items():
        print(f"{product:<40} {count:<10}")
        results.append((product, count))
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for product, count in results:
                f.write(f"{product}\t{count}\n")
        print(f"\nThe results have been saved to {output_file}")
    return product_counts

if __name__ == "__main__":
    csv_file_path = "data/valid/CDs_and_Vinyl_5_2015-10-2018-11.csv" 
    txt_file_path = "data/info/CDs_and_Vinyl_5_2015-10-2018-11.txt"
    output_file_path = "count_result/CDs_and_Vinyl/valid_count_results.txt" 
    count_product_occurrences(csv_file_path, txt_file_path, output_file_path)