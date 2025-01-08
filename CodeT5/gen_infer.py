import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def add_line_numbers(code_snippet):
    """
    Add line numbers to a code snippet.
    """
    lines = code_snippet.strip().split('\n')
    return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])

def preprocess_function(examples):
    """
    Preprocess code snippet and locations for tokenization.
    """
    inputs = [add_line_numbers(doc) for doc in examples["function"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    return model_inputs

def is_shorter_than_max_length(example):
    text_column_name = "function"
    tokens = tokenizer.tokenize(example[text_column_name])
    return len(tokens) < max_input_length
def transform_to_binary_labels(line_numbers, total_lines):
    """
    Transform line numbers to binary labels for sequence labeling.
    """
    labels = ['O'] * total_lines
    for line in line_numbers:
        if line <= total_lines:
            labels[line - 1] = 'B-LOC'
    return labels

def predict_location(code_snippet, model, tokenizer, max_input_length):
    """
    Generate predictions for a given code snippet.
    """
    input_text = add_line_numbers(code_snippet)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    output_sequences = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_target_length)
    decoded_prediction = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return [int(r) for r in re.findall(r'(\d+):', decoded_prediction)]

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load tokenizer and model
model_id = 'Salesforce/codet5-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained("codet5-base-gen/checkpoint-10810").to(device)
max_input_length = 512
max_target_length = 256

# Load and preprocess test dataset
test_dataset = load_dataset("csv", data_files="../data/test.csv", split="train")

# # Filter the dataset
# filtered_test_dataset = test_dataset.filter(is_shorter_than_max_length)
# print(f"Original size: {len(test_dataset)}, Filtered size: {len(filtered_test_dataset)}")
# test_dataset = filtered_test_dataset
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Inference and transformation
predictions = [predict_location(item['function'], model, tokenizer, max_input_length) for item in tqdm(test_dataset)]
predicted_binary_labels = [transform_to_binary_labels(pred, len(item['function'].split('\n'))) for pred, item in zip(predictions, test_dataset)]
true_binary_labels = [transform_to_binary_labels(eval(item['location']), len(item['function'].split('\n'))) for item in test_dataset]

# Load seqeval from evaluate library
seqeval_metric = evaluate.load("seqeval")

# Compute metrics using seqeval from evaluate
results = seqeval_metric.compute(predictions=predicted_binary_labels, references=true_binary_labels)

print("Precision:", results["overall_precision"])
print("Recall:", results["overall_recall"])
print("F1 Score:", results["overall_f1"])
print("Accuracy:", results["overall_accuracy"])
