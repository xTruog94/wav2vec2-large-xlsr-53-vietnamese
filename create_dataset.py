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
    audio["array"] = librosa.resample(array[0], 48_000, 16_000)
    audio["sampling_rate"] = sampling_rate
    audio["path"] = path
    return audio

def prepare_dataset(batch,processor):
    res = {}
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    res["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"],padding=True).input_values[0].tolist()
    # res["input_length"] = len(res["input_values"])
    with processor.as_target_processor():
        res["labels"] = processor(batch["text"]).input_ids
    return res

def to_dataloader(*data,batch_size):
    print(data)
    train_data = TensorDataset(data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def create_data(wav_folder,text_folder,folder_save):

    assert os.path.exists(wav_folder) and os.path.exists(text_folder)
    
    data = []
    train_size = 0.7
    test_size = 0.3
    list_file = os.listdir(wav_folder)
    vocal = {}
    all_text = []
    count =0 
    vocab_file = args.model_path+'vocab.json'

    for f in tqdm(list_file[:100]):
        tmp = {}
        file_name = get_file_name(f)

        with open(text_folder+file_name+'.txt','r') as fp:
            text = fp.read()
        
        text = clean(text)
        path = wav_folder + f

        tmp['audio'] = speech_file_to_array_fn(path)
        tmp['text'] = text
        all_text.append(text)
        data.append(tmp)

    create_vocab(all_text,vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(args.model_path)

    data_cleaned = []
    input_values = []
    lbs = []
    for batch in tqdm(data):
        batch_cleaned = prepare_dataset(batch,processor)
        ## batch_cleaned = prepare_data(resample,prepare_dataset,batch=batch)
        # data_cleaned.append(batch_cleaned)
        input_values.append(batch_cleaned["input_values"])
        lbs.append(batch_cleaned["labels"])
    

    length = len(input_values)
    train_input = input_values[0:int(train_size*length)]
    train_lb = lbs[0:int(train_size*length)]

    test_input = input_values[int(train_size*length):]
    test_lb = lbs[int(train_size*length):]

    # df_train = pd.DataFrame(data=train_data,columns=["input_values","labels"])
    # df_test = pd.DataFrame(data=test_data,columns=["input_values","labels"])
    # df_train.to_csv(folder_save+"data_train.csv",index=False)
    # df_test.to_csv(folder_save+"data_test.csv",index=False)

    # train_dataloader = to_dataloader(train_input,train_lb,batch_size=batch_size)
    # test_dataloader = to_dataloader(test_input,test_lb,batch_size=batch_size)
    # cached_features_file = args.cached_save
    # torch.save(train_dataloader,cached_features_file+"cache-train-{}".format(batch_size))
    # torch.save(test_dataloader,cached_features_file+"cache-test-{}".format(batch_size))
    # return train_dataloader,test_dataloader
    train_data = {"input_values":train_input,"labels":train_lb}
    test_data = {"input_values":test_input,"labels":test_lb}
    cached_features_file = args.cached_save
    to_json(cached_features_file + 'train_data.json', train_data)
    to_json(cached_features_file + 'test_data.json', test_data)
    
    return train_data,test_data
if __name__ == "__main__":

    create_data(args.data_sounds, args.data_transcripts, args.model_path)