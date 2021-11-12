import torch
import torch.nn as nn
from transformers import Trainer,TrainingArguments,Wav2Vec2CTCTokenizer,Wav2Vec2Processor,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
from utils import *
import torchaudio
import json
import pickle
from arguments import load_arg
from datasets import load_dataset, load_metric
import numpy as np
from datasets import Dataset
args = load_arg()
from DataCollator import DataCollatorCTCWithPadding
import torch
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
    
from create_dataset import create_data

def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        print(pred_ids.shape,pred_ids)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return wer

if __name__ == "__main__":
    #load data
    cached_features_file  = args.cached_save
    if os.path.exists(cached_features_file+"train_data.json") and \
        os.path.exists(cached_features_file+"test_data.json") and \
        args.use_cache:
        print("loading data train from cache...")
        train_dataloader = open_json(cached_features_file+"train_data.json")
        test_dataloader = open_json(cached_features_file+"test_data.json")
    else:
        train_dataloader, test_dataloader = create_data(args.data_sounds, args.data_transcripts, args.model_path)
    
    
    tokenizer = Wav2Vec2CTCTokenizer(args.model_path+'vocab.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    data_train = Dataset.from_dict(train_dataloader)
    data_test = Dataset.from_dict(test_dataloader)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")
    
    model = Wav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-large-xlsr-53", 
                    attention_dropout=0.1,
                    hidden_dropout=0.1,
                    mask_time_prob=0.05,
                    ctc_loss_reduction="mean", 
                    pad_token_id=processor.tokenizer.pad_token_id,
                    vocab_size=len(processor.tokenizer),
                    mask_time_length=8
    )
    print(processor.tokenizer.pad_token_id)
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
    # print(model)
    training_args = TrainingArguments(
            output_dir=args.model_path,
            group_by_length=False,
            per_device_train_batch_size=1,
            evaluation_strategy="steps",
            num_train_epochs=args.epochs,
            fp16=True,
            gradient_checkpointing=True,
            save_steps=len(data_train),
            eval_steps=len(data_train),
            logging_steps=len(data_train),
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=len(data_train),
            save_total_limit=2,
            no_cuda = args.no_cuda
            )

    print(data_collator)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    
