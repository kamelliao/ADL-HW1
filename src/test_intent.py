import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tkinter import W
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import SeqClsDataset
from model import SeqClassifier, CNNClassifier
from utils import Vocab
import parsers

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset('test', data, vocab, intent2idx, args.max_len)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    if args.model == 'CNN':
        model = CNNClassifier(
            embeddings,
            args.dropout,
            dataset.num_classes
        )
    else:
        model = SeqClassifier(
            embeddings,
            args.model,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.bidirectional,
            dataset.num_classes,
            args.embed_type,
        )
    model.eval()
    model_weight = torch.load(args.model_file)
    model.load_state_dict(model_weight)

    # TODO: predict dataset
    result = {'id': [], 'intent': []} 
    with torch.no_grad():
        for (data, doc_id) in tqdm(test_loader):
            pred = model(data)
            labels = pred.argmax(dim=1).tolist()
            result['id'].extend(doc_id)
            result['intent'].extend([dataset.idx2label(label) for label in labels])

    # TODO: write prediction to file (args.pred_file)
    pd.DataFrame(result).to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    dirs = parsers.parser_dirs('intent')
    base = parsers.parser_base()

    parser = ArgumentParser(parents=[dirs, base])
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")
    parser.add_argument("--model_file", type=Path, default="./ckpt/intent/best.pt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
