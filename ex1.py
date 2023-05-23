from transformers import BertTokenizer, BertForSequenceClassification, \
    RobertaForSequenceClassification, ElectraForSequenceClassification

from dataclasses import dataclass, field
import random
import numpy as np
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    ElectraTokenizer,
    RobertaTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
import sacrebleu
from evaluate import load
from sklearn.metrics import accuracy_score
import time
import sacrebleu
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# wandb.login(key='b681ba776753b19ecb73cddd77dddce2d67036f5')
# wandb.init(project='ex1-try1')


def set_seed(seed: int):
    """
    Set the seed value for randomization in order to ensure reproducibility.

    Args:
        seed (int): The seed value to set.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(p: EvalPrediction):
    """
    Compute evaluation metrics for a classification task based on the predicted and true labels.

    Args:
        p (EvalPrediction): The evaluation prediction object containing predicted and true labels.

    Returns:
        metrics (dict): A dictionary containing the computed metrics. In this case, it includes the accuracy.

    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = accuracy_score(preds, p.label_ids)
    return {"accuracy": result}

def train_and_evaluate(training_args, dataset, model_name: str,
                       seeds,
                       train_samples: int,
                       val_samples: int):
    """
    Train and evaluate a sequence classification model using the provided training arguments and dataset.

    Args:
        training_args: The training arguments for fine-tuning the model.
        dataset (Dataset): The dataset for training and evaluation.
        model_name (str): The name of the pre-trained model to use.
        seeds (list): A list of seed values for randomization during training.
        train_samples (int): The number of training samples to include. Set to -1 for all samples.
        val_samples (int): The number of validation samples to include. Set to -1 for all samples.

    Returns:
        mean_accuracy (float): The mean accuracy across different seeds.
        std_dev_accuracy (float): The standard deviation of accuracy across different seeds.
        best_seed: The seed value that achieved the highest accuracy.
        best_model: The model with the highest accuracy.
        best_tokenizer: The tokenizer associated with the best model.

    """
    best_seed, best_model, best_tokenizer = None, None, None
    best_acc = 0.0
    acc = []
    for seed in seeds:
        set_seed(seed)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, truncation=True)
        def tokenize_function(example):
            return tokenizer(example['sentence'], padding='max_length', truncation=True)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        # Prepare the data for fine-tuning
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        if train_samples != -1:
            train_dataset = train_dataset.select(range(train_samples))
        if val_samples != -1:
            eval_dataset = eval_dataset.select(range(val_samples))
        # Set up the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        # Fine-tune the model
        trainer.train()
        # Evaluate the model
        eval_results = trainer.evaluate()
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        model.eval()
        acc.append(metrics["eval_accuracy"])
        if best_acc <= metrics["eval_accuracy"]:
            best_acc = metrics["eval_accuracy"]
            best_seed = seed
            best_model = model
            best_tokenizer = tokenizer
    mean_accuracy = np.mean(acc)
    # Compute standard deviation of accuracy
    std_dev_accuracy = np.std(acc)
    return mean_accuracy, std_dev_accuracy, best_seed, best_model, \
           best_tokenizer

def create_predict_file(model, tokenizer,
                        test_samples: int):
    """
    Create a prediction file using the provided model and tokenizer.

    Args:
        model (PreTrainedModel): The pre-trained model for prediction.
        tokenizer (PreTrainedTokenizer): The tokenizer for tokenizing the text.
        test_samples (int): The number of test samples to include. Set to -1 for all samples.

    Returns:
        None

    """
    test_dataset = load_dataset("glue", "sst2", split="test")
    if test_samples != -1:
        test_dataset = test_dataset.select(range(test_samples))
    test_texts = test_dataset["sentence"]

    # Open the file to write predictions
    with open("predictions.txt", "w") as file:
        for text in test_texts:
            # Tokenize the text
            inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
            inputs = {key: value.to('cuda') for key, value in inputs.items()}

            # Perform the prediction
            with torch.no_grad():
                outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits).item()

            # Write the prediction to the file
            file.write(f"{text}###{predicted_label}\n")


def arg_parse():
    """
    Parse command-line arguments for training.

    Returns:
        tuple: A tuple containing the parsed arguments and training arguments.

    """
    parser = argparse.ArgumentParser()
    training_args = TrainingArguments(output_dir='./results')
    # Add positional arguments in the desired order
    parser.add_argument('number_of_seeds', type=int)
    parser.add_argument('train_samples', type=int)
    parser.add_argument('val_samples', type=int)
    parser.add_argument('test_samples', type=int)
    # Parse the command-line arguments
    args = parser.parse_args()
    return args, training_args


if __name__ == '__main__':
    """
    Main function for training and evaluating multiple models on the SST-2 dataset.
    """
    args, training_args = arg_parse()
    raw_datasets = load_dataset("glue", "sst2")
    models_names = ['roberta-base','google/electra-base-generator',
                    'bert-base-uncased']
    seeds = np.random.randint(50, size=args.number_of_seeds)
    best_model_name = None
    best_acc, best_seed = 0.0, 0.0
    best_model = None
    best_tokenizer = None
    start_time_train = time.time()
    with open("res.txt", "w") as file_res:
        for model_name in models_names:
            print(model_name)
            mean_accuracy, std_dev_accuracy, seed, model, tokenizer = train_and_evaluate(
                training_args, raw_datasets, model_name, seeds, args.train_samples,
                args.val_samples)
            file_res.write(f"{model_name},{mean_accuracy} +- {std_dev_accuracy}\n")
            if best_acc <= mean_accuracy:
                best_acc, best_seed, best_model_name, best_model, \
                best_tokenizer = mean_accuracy, seed, model_name, model, tokenizer
        end_time_train = time.time()
        runtime_train = end_time_train - start_time_train
        file_res.write(f"----\n")
        file_res.write(f"train time,{runtime_train}\n")
        start_time_test = time.time()
        create_predict_file(best_model, best_tokenizer, args.test_samples)
        end_time_test = time.time()
        runtime_test = end_time_test - start_time_test
        file_res.write(f"predict time,{runtime_test}\n")
        print("doneee")