import glob
from pathlib import Path
import os
import random
import re
from typing import List

from fire import Fire
from moviepy.editor import concatenate, VideoFileClip
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess
import sys
import torch
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, GlueDataset
from transformers import TrainingArguments, Trainer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.processors.utils import SingleSentenceClassificationProcessor, InputFeatures
import youtube_dl

from main import clean_up_string, convert_time

training_args = TrainingArguments(
    output_dir="./models/albert-base-v2",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=32,
    per_gpu_eval_batch_size=32,
    num_train_epochs=1,
    logging_steps=5,
    logging_first_step=True,
    save_steps=1000,
    evaluate_during_training=True,
)

def main():
    
    # Initializing a pre-trained ALBERT-base style
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    # Initialize data iterators
    train_generator = SingleSentenceClassificationProcessor()
    train_generator.add_examples_from_csv(file_name='data/train.tsv', column_label=1, column_text=0)
    train_dataset = train_generator.get_features(tokenizer=tokenizer)#, return_tensors='pt')

    eval_generator = SingleSentenceClassificationProcessor()
    eval_generator.add_examples_from_csv(file_name='data/dev.tsv', column_label=1, column_text=0)
    eval_dataset = train_generator.get_features(tokenizer=tokenizer)#, return_tensors='pt')

    test_generator = SingleSentenceClassificationProcessor()
    test_generator.add_examples_from_csv(file_name='data/test.tsv', column_label=1, column_text=0)
    test_dataset = train_generator.get_features(tokenizer=tokenizer)#, return_tensors='pt')
 
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    test_batch = next(iter(test_dataset))
    print(f'Test batch is {test_batch}')
    pred = model(torch.tensor(test_batch.input_ids).unsqueeze(0).cuda(), torch.tensor(test_batch.label).unsqueeze(0).cuda())
    print(f'Prediction: {pred}')

if __name__ == '__main__':
    main()