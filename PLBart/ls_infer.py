from transformers import AutoTokenizer
from modeling_plbart import PLBartForTokenClassification
from tqdm import tqdm
import evaluate
import torch
import json


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained('./my_awesome_ds_model/checkpoint-10810/')
model = PLBartForTokenClassification.from_pretrained('./my_awesome_ds_model/checkpoint-10810/', device_map="auto")
id2label = {0: 'O', 1: 'B-LOC'}
with open(f'../data/test.jsonl', 'r') as reader, open('ls_plbart.out', 'w') as res:
    true_labels = []
    true_predictions = []
    for line in tqdm(reader):
        row = json.loads(line)

        # Tokenize the entire code
        inputs = tokenizer(row['tokens'], is_split_into_words=True, return_tensors='pt', truncation=True).to(device)
        labels = [id2label[t] for t in row['ner_tags']]
        # Run the model
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)[0]

        predicted_labels = []
        ground_labels = []

        length = len(inputs['input_ids'][0])
        # Tokenize each word separately and keep track of their indices
        tokenized_words = [tokenizer.tokenize(word) for word in row['tokens']]

        pre_idx = -1
        for word_index, word_tokens in enumerate(tokenized_words):
            word_len = len(word_tokens)
            if word_len:
                ground_labels.append(labels[word_index])
                if pre_idx+word_len < length:
                    pre_idx += word_len
                    p_id = predictions[pre_idx].item()
                    predicted_labels.append(id2label[p_id])


        assert len(predicted_labels) <= len(ground_labels)
        predicted_labels += ['O']*(len(ground_labels)-len(predicted_labels))
        res.write(repr(predicted_labels)+'\n')

        true_labels.append(ground_labels)
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


