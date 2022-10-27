import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import string

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
import parsers


def unpad_batch(target: List[List[Any]], original_len: List[int]):
    return [seq[:seq_len] for seq, seq_len in zip(target, original_len)]


def main(args):
    torch.manual_seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2tag: Dict[int, str] = {value: key for key, value in tag2idx.items()}

    # character level info
    char_vocab = Vocab(string.ascii_lowercase + string.digits + string.punctuation)
    char_embedding = torch.rand(len(char_vocab), 25)

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(char_vocab, 'test', data, vocab, tag2idx, args.max_len)
    test_loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        char_embedding,
        embeddings,
        args.model,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.embed_type
    )
    model_weight = torch.load(args.model_file)
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    result = {'id': [], 'tags': []} 
    with torch.no_grad():
        for (data, data_char, doc_id, original_len) in tqdm(test_loader):
            pred = model(data, data_char).transpose(1, 2)
            labels: List[List[int]] = pred.argmax(dim=1).tolist()
            labels: List[List[int]] = unpad_batch(labels, original_len)
            labels: List[List[str]] = test_loader.dataset.idx2label_batch(labels)
            
            result['id'].extend(doc_id)
            result['tags'].extend([" ".join(label) for label in labels])

    pd.DataFrame(result).to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    dirs = parsers.parser_dirs('slot')
    base = parsers.parser_base()

    parser = ArgumentParser(parents=[dirs, base])
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")
    parser.add_argument("--model_file", type=Path, default="./ckpt/slot/best.pt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)