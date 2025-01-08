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

    all_predictions = []
    for i in range(0, total_length, step):
        # Prepare the input batch
        batch = {k: v[:, i:i+max_length] for k, v in tokens.items()}
        with torch.no_grad():
            # Run the model
            logits = model(**batch).logits

        # Convert logits to label predictions
        predictions = torch.argmax(logits, dim=2)[0]
        all_predictions.extend(predictions[:step])

    predicted_labels = []
    previous_word_idx = None
    last_token_position = None

    for word_idx_position, (word_idx, prediction) in enumerate(zip(word_ids, all_predictions)):
        if word_idx is not None:
            # Mark the position of the last token of the current word.
            last_token_position = word_idx_position
        if word_idx != previous_word_idx and previous_word_idx is not None:
            # Update the label for the last token of the previous word.
            p_id = all_predictions[
                last_token_position].item()  # Get the prediction for the last token of the previous word.
            predicted_labels.append(id2label[p_id])
        previous_word_idx = word_idx

    # Ensure the last word's label is also updated
    if last_token_position is not None and previous_word_idx is not None:
        p_id = all_predictions[last_token_position].item()
        predicted_labels.append(id2label[p_id])
    predicted_labels += ['O'] * (len(labels) - len(predicted_labels))

    return predicted_labels


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "my_awesome_ds_model/checkpoint-10810"
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


