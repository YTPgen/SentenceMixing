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
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.processors.utils import InputFeatures
import youtube_dl

from main import clean_up_string, convert_time


ydl_opts = {
    'skip_download': True,
    'subtitleslangs': ['en'],
    'writesubtitles': True,
    'writeautomaticsub': True, 
    'subtitlesformat': 'vtt',
}

def download_captions(urls: List):
    # download captions from youtube
    # automatic captioning needs to be enabled by the uploader
    # if no subs are available then simply skip this video
    for url in urls:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


def download_data(data_type: str):
    try:
        with open(data_type+'_urls.txt') as f:
            urls = f.readlines()
    except IOError:
        print(f'Requires a file named {data_type}_urls.txt in this dir. containing urls for youtube videos')
    urls = [x.strip() for x in urls]
    download_captions(urls)


def pre_process_data():
    file_names = []
    for sub_file in glob.glob('*.vtt'):
        file_names.append(sub_file)    
    
    times_texts = []
    for file in file_names:
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            current_text =  line.replace('\n','')
            current_text = re.findall('^([A-Za-z|\W]+)', current_text)
            if not current_text:
                continue
            else:
                current_text = [clean_up_string(x.strip(' ')) for x in current_text if x][0]
                if len(current_text.split(' ')) != 1:
                    times_texts.append(current_text)

    # Remove copies
    texts = set(times_texts)
    return texts


def remove_vtt_files():
    command = 'rm *.vtt'
    subprocess.call(command, shell=True)


def create_dataset(data_type: str, label: int):
    download_data(data_type)
    data = pre_process_data()
    data_w_labels = []
    for sentence in data:
        data_w_labels.append([sentence, label])
    remove_vtt_files()
    return data_w_labels


def main():
    if not Path(os.getcwd()+'/train.tsv').is_file():
        ytp_data = create_dataset('ytp', label=1)
        source_data = create_dataset('source', label=0)
    else:
        return
    
    dataset = ytp_data + source_data
    dataset = pd.DataFrame(dataset, columns=['sentence','label'])

    print(dataset.query('label==1'))
    print(dataset.query('label==0'))

    train, test = train_test_split(dataset, test_size=0.1) 
    dev, test = train_test_split(test, test_size=0.5)
    
    train.to_csv('train.tsv', sep='\t', index=False, header=False)
    dev.to_csv('dev.tsv', sep='\t', index=False, header=False)
    test.to_csv('test.tsv', sep='\t', index=False, header=False)


if __name__ == '__main__':
    main()