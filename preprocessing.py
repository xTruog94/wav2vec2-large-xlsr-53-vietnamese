import torch
import os
import itertools
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re
import json
from collections import Counter
from utils import *
import torchaudio
from arguments import load_arg
import librosa
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from tqdm import tqdm
import pandas as pd
import math
args = load_arg()

def create_vocab(text,fp):
    text = " ".join(text)
    vocab_all = list(dict(Counter(text)).keys())
    vocab = {v:k for k,v in enumerate(vocab_all)}
    vocab['|'] = vocab[" "]
    
    vocab.pop(" ")
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    to_json(fp,vocab)

def speech_file_to_array_fn(path):
    audio = {}
    speech_array, sampling_rate = torchaudio.load(path)
    array = speech_array.numpy().astype(float)
    audio["array"] = librosa.resample(array[0], 48_000, 16_000).tolist()
    audio["sampling_rate"] = sampling_rate
    audio["path"] = path
    return audio

def split_dataset(files,audios,texts,train_size=0.7):
    length = math.ceil(len(files)*train_size)
    print(length)
    train_data = {'file':files[:length], \
                'audio':audios[:length],\
                'text':texts[:length]}
    test_data = {'file':files[length:], \
                'audio':audios[length:],\
                'text':texts[length:]}
    return train_data, test_data

def run(wav_folder,text_folder,folder_save):
    
    assert os.path.exists(wav_folder) and os.path.exists(text_folder)
    data = []
    list_file = os.listdir(wav_folder)
    vocal = {}
    count =0 
    vocab_file = args.model_path+'vocab.json'
    files = []
    audios = []
    texts = []
    for f in tqdm(list_file):
        tmp = {}
        file_name = text_folder+get_file_name(f)+'.txt'
        with open(file_name,'r') as fp:
            text = fp.read()
        
        text = clean(text)
        path = wav_folder + f
        files.append(path)
        audios.append(speech_file_to_array_fn(path))
        texts.append(text)
    create_vocab(texts,vocab_file)
    train_data, test_data = split_dataset(files,audios,texts)
    cached_features_file = args.cached_save
    to_json(cached_features_file + 'train_full.json', train_data)
    to_json(cached_features_file + 'test_full.json', test_data)

if __name__ == "__main__":
    run(args.data_sounds, args.data_transcripts, args.model_path)