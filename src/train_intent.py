import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb

from dataset import SeqClsDataset
from model import SeqClassifier, CNNClassifier
from utils import Vocab, gradient_norm
import parsers

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # wandb.init(project='adl-hw1-intent-report', entity='kamelliao')
    # wandb.config.update(args)

    torch.manual_seed(args.seed)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split, split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn, pin_memory=True)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn, pin_memory=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    if args.model == 'CNN':
        model = CNNClassifier(
            embeddings,
            args.dropout,
            len(intent2idx)
        ).to(args.device)
    else:
        model = SeqClassifier(
            embeddings,
            args.model,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.bidirectional,
            len(intent2idx),
            args.embed_type,
        ).to(args.device)

    if args.load_model:
        model_weight = torch.load(args.load_model)
        model.load_state_dict(model_weight)   

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(args.device), target.to(args.device)

            pred = model(data)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            # norm = gradient_norm(model)
            # wandb.log({'gradient_norm': norm})
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, 2)
            optimizer.step()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        correct = 0
        dev_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dev_loader)):
                data, target = data.to(args.device), target.to(args.device)

                pred = model(data)
                loss = loss_fn(pred, target)
                dev_loss += loss.item()
                correct += (pred.argmax(dim=1) == target).sum().item()
        
        dev_loss /= len(dev_loader.dataset)
        dev_acc = 100. * correct / len(dev_loader.dataset)
        print(f'Epoch {epoch+1:3d} | dev loss {dev_loss:.5f} | dev acc {correct:>4d}/{len(dev_loader.dataset)} ({dev_acc:.1f}%)')

        # wandb.log({'dev_loss': dev_loss, 'dev_acc': dev_acc})

        # save model 
        torch.save(model.state_dict(), args.ckpt_dir.joinpath('model.pt'))

    # TODO: Inference on test set


def parse_args() -> Namespace:
    dirs = parsers.parser_dirs('intent')
    base = parsers.parser_base()

    parser = ArgumentParser(parents=[dirs, base])
    parser.add_argument("--load_model", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.add_argument("--num_epoch", type=int, default=50)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir = args.ckpt_dir.joinpath(args.model_id)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
