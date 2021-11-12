from datasets import load_dataset
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path_to_ngram", type=str, required=True, help="Path to kenLM ngram"
)
args = parser.parse_args()

ds = load_dataset("multilingual_librispeech", f"{args.language}", split="train")

with open("text.txt", "w") as f:
    f.write(" ".join(ds["text"]))

os.system(f"./kenlm/build/bin/lmplz -o 5 <text.txt > {args.path_to_ngram}")