from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "./cross_ls_model/checkpoint-12370/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)
id2label = {0: 'O', 1: 'B-LOC'}
with open(f'../cross_data/test.jsonl', 'r') as reader, open('cross_ls.txt', 'w') as res:
    true_labels = []
    true_predictions = []
    for line in reader:
        row = json.loads(line)

        # Tokenize the entire code
        inputs = tokenizer(row['tokens'], is_split_into_words=True, return_tensors='pt', truncation=True).to(device)
        labels = [id2label[t] for t in row['ner_tags']]
        # Run the model
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)


        word_ids = inputs.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        predicted_labels = []
        for word_idx, prediction in zip(word_ids, predictions[0]):
            if word_idx is None:
                pass
            elif  word_idx != previous_word_idx:
                p_id = prediction.item()
                predicted_labels.append(id2label[p_id])
            previous_word_idx = word_idx
        predicted_labels += ['O']*(len(labels)-len(predicted_labels))
        res.write(repr(predicted_labels)+'\n')

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


