import pandas as pd
import re
import evaluate
import json
from tqdm import tqdm

def process_cwe_data(files):
    # Combine datasets
    combined_df = pd.concat([pd.read_csv(file) for file in files])

    # Normalize 'cwe' values
    combined_df['cwe'] = combined_df['cwe'].apply(lambda x: 'CWE-UNK' if not re.match(r'^CWE-\d+$', str(x)) else x)

    # Count occurrences of each 'cwe'
    cwe_counts = combined_df['cwe'].value_counts()

    # Print top 10
    return cwe_counts.head(10).index.tolist()

def analyze_top_cwe_and_extract_ids(csv_path):
    df = pd.read_csv(csv_path)
    df['cwe'] = df['cwe'].apply(lambda x: x if pd.isnull(x) or x.startswith('CWE-') and x[4:].isdigit() else 'CWE-UNK')

    for cwe in top_10_cwes:
        line_numbers = df.index[df['cwe'] == cwe].tolist()
        print(f"CWE Type: {cwe}")

        true_labels = []
        true_predictions = []

        with open(f'../data/test.jsonl', 'r') as reader:
            for i, line in enumerate(tqdm(reader)):
                if i in line_numbers:
                    row = json.loads(line)
                    labels = [id2label[t] for t in row['ner_tags']]
                    true_labels.append(labels)

        with open('ls_codellama.out', 'r') as res:
            for i, line in enumerate(tqdm(res)):
                if i in line_numbers:
                    predicted_labels = eval(line.strip())
                    true_predictions.append(predicted_labels)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        ret = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        print(ret)




# Specify the paths to the files
files = ['../data/train.csv', '../data/valid.csv', '../data/test.csv']

# Execute the function
top_10_cwes = process_cwe_data(files)
id2label = {0: 'O', 1: 'B-LOC'}
seqeval = evaluate.load("seqeval")
analyze_top_cwe_and_extract_ids("../data/test.csv")


