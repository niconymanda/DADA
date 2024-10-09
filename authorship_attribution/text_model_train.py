import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import argparse

from model import AuthorshipAttributionLLM


def init_env():
    seed_val = 42
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a text classification model')
    parser.add_argument('--data', type=str, default='~/DADA/Data/WikiQuotes_final.csv', help='Path to the input data file')
    parser.add_argument('--epochs', type=int, default=35, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model', type=str, default='FacebookAI/roberta-base', help='Model to use')
    return parser.parse_args()

def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    
    data['label'] = data['label'].astype(int)
    label_counts = data['label'].value_counts()
    labels_to_keep = label_counts[label_counts >= 250].index
    data = data[data['label'].isin(labels_to_keep)]
    print(f"Number of authors that have more then 100 quotes: {len(data['label'].unique())}")
    
    authors = data['author_name'].unique()
    for i, author in enumerate(authors):
        data.loc[data['author_name'] == author, 'label'] = i
        
    spoofed_data = data[data['type'] == 'spoof']
    data = data[data['type'] != 'spoof']

    return data, spoofed_data

def prepare_datasets(data, spoofed_data,parser):
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['label'], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.6, stratify=temp_data['label'], random_state=42)

    train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_data[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'label']])
    spoofed_data = Dataset.from_pandas(spoofed_data[['text', 'label']])
    

    tokenizer = AutoTokenizer.from_pretrained(parser.model, pad_token='<|pad|>')
    # if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    spoofed_data = spoofed_data.map(tokenize, batched=True, batch_size=len(spoofed_data))
    
    
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    spoofed_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    class_names = data['author_name'].unique()
    global id2label
    id2label = {i: label for i, label in enumerate(class_names)}
    
    return train_dataset, val_dataset, test_dataset, spoofed_data

def train_model(train_dataset, val_dataset, parser, num_classes):
    print("Training model")
    print(f"Number of classes: {num_classes}")
    config = AutoConfig.from_pretrained(parser.model)
    config.update({"id2label": id2label})
    model = AuthorshipAttributionLLM(parser.model, num_classes)   
    repository_id = f"./output/{parser.model}_{parser.epochs}"
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=repository_id,
        num_train_epochs=parser.epochs,
        per_device_train_batch_size=parser.batch_size,
        per_device_eval_batch_size=parser.batch_size,
        eval_strategy="epoch",
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=parser.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="tensorboard",
        metric_for_best_model="eval_loss", 
        greater_is_better=False 
    )
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(f"{repository_id}/model")
    return trainer

def evaluate_model(model, test_dataset):
    print("Evaluating model")
    predictions = model.predict(test_dataset)
    y_true = predictions.label_ids
    y_pred = predictions.predictions
    y_pred = y_pred[0].argmax(axis=-1)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    classes = np.unique(y_true)
    y_true_binarized = label_binarize(y_true, classes=classes)
    y_pred_binarized = label_binarize(y_pred, classes=classes)
    auc = roc_auc_score(y_true_binarized, y_pred_binarized, average="macro", multi_class="ovr")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"ROC AUC score: {auc}")
    results = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc_score': auc}
    return results
       
def write_results_to_file(results, file_path, parser):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Writing results to file {file_path}")
    with open(file_path, 'a') as f: 
        f.write(f"Model: {parser.model}, epochs {parser.epochs}, batch_size {parser. batch_size}, learning rate {parser.learning_rate}\n")
        f.write("Accuracy, Precision, Recall, F1, AUC\n")
        f.write(f"{results['accuracy']}, {results['precision']}, {results['recall']}, {results['f1']}, {results['roc_auc_score']}\n")
        
def load_trainer_from_path(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def main():
    init_env()
    parser = parse_arguments()
    data, spoofed_data = load_data(parser.data)
    train_dataset, val_dataset, test_dataset, spoofed_test_dataset = prepare_datasets(data, spoofed_data, parser)
    num_classes = len(data['label'].unique())
    model = train_model(train_dataset, val_dataset, parser, num_classes)
    print("Evaluating model on test data")
    results = evaluate_model(model, test_dataset)
    file = f"./output/results.txt"
    write_results_to_file(results, file, parser)
    
    print("Evaluating model on spoofed data")
    results_sp = evaluate_model(model, spoofed_test_dataset)
    write_results_to_file(results_sp, f"./output/spoofed_results.txt", parser)
    
if __name__ == "__main__":  
    main()
    
