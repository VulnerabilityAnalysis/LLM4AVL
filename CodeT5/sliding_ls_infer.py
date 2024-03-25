from transformers import AutoTokenizer
from tqdm import tqdm
from modeling_codet5 import T5ForTokenClassification
import evaluate
import torch
import json
import torch


def sliding_window_predict(row, tokenizer, model, id2label, max_length):
    # Tokenize the text
    tokens = tokenizer(row['tokens'], is_split_into_words=True, return_tensors='pt', truncation=False).to(device)
    total_length = tokens.input_ids.size(1)
    step = max_length // 2  # Overlap size for sliding window

    # Get the word IDs for the entire sequence
    word_ids = tokens.word_ids(batch_index=0)

    predicted_labels = [None] * len(row['tokens'])  # Initialize with None for each word in the original text
    for i in range(0, total_length, step):
        # Prepare the input batch
        batch = {k: v[:, i:i+max_length] for k, v in tokens.items()}
        with torch.no_grad():
            # Run the model
            logits = model(**batch).logits

        # Convert logits to label predictions
        predictions = torch.argmax(logits, dim=2)[0]

        previous_word_idx = None
        for token_idx, prediction in enumerate(predictions):
            # Calculate the actual word index in the original text
            word_idx = word_ids[i + token_idx] if i + token_idx < len(word_ids) else None
            if word_idx is None or word_idx >= len(predicted_labels):
                continue
            if word_idx != previous_word_idx:
                p_id = prediction.item()
                if predicted_labels[word_idx] is None:  # Fill the label only if it's not already filled
                    predicted_labels[word_idx] = id2label[p_id]
            previous_word_idx = word_idx

    predicted_labels = [label if label is not None else 'O' for label in predicted_labels]
    return predicted_labels

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "sliding-ls-model/checkpoint-48740"
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
model = T5ForTokenClassification.from_pretrained(model_path).to(device)
id2label = {0: 'O', 1: 'B-LOC'}
with open(f'../data/test.jsonl', 'r') as reader:
    true_labels = []
    true_predictions = []
    for line in tqdm(reader):
        row = json.loads(line)

        # Tokenize the entire code
        inputs = tokenizer(row['tokens'], is_split_into_words=True, return_tensors='pt', truncation=True)
        labels = [id2label[t] for t in row['ner_tags']]
        max_length = tokenizer.model_max_length
        predicted_labels = sliding_window_predict(row, tokenizer, model, id2label,
                                                  max_length=max_length)
        assert len(predicted_labels) == len(labels)
        # predicted_labels += ['O'] * (len(labels) - len(predicted_labels))

        true_labels.append(labels)
        true_predictions.append(predicted_labels)

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    ret =  {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    print(ret)


