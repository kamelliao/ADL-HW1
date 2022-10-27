import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import string

from seqeval.scheme import IOB2
from seqeval.metrics import classification_report
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, gradient_norm, accuracies
import parsers

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    wandb.init(project='adl-hw1-slot-report', entity='kamelliao')
    wandb.config.update(args)

    # set random seed
    torch.manual_seed(args.seed)

    # load 'vocab.pkl' and 'tag2idx.json'
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # character level info
    char_vocab = Vocab(string.ascii_lowercase + string.digits + string.punctuation)
    char_embedding = torch.rand(len(char_vocab), 25) - 0.5

    # load datasets
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(char_vocab, split, split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # build dataloaders
    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn, pin_memory=True)
    dev_loader = DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[DEV].collate_fn, pin_memory=True)

    # load GloVe pre-trained word embedding
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # define model, optimizer, and loss function
    model = SeqTagger(
        char_embedding,
        embeddings,
        args.model,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag2idx),
        args.embed_type
    ).to(args.device)
    
    if args.load_model:
        model_weight = torch.load(args.load_model)
        model.load_state_dict(model_weight)  

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # start training
    for epoch in range(args.num_epoch):
        # training loop
        model.train()
        for batch_idx, (data, data_char, target) in enumerate(tqdm(train_loader)):
            data, data_char, target = data.to(args.device), data_char.to(args.device), target.to(args.device)

            pred = model(data, data_char).transpose(1, 2)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            # norm = gradient_norm(model)
            # wandb.log({'gradient_norm': norm})
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, 2)
            optimizer.step()

        # evaluation loop
        model.eval()
        dev_loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (data, data_char, target) in enumerate(tqdm(dev_loader)):
                data, data_char, target = data.to(args.device), data_char.to(args.device), target.to(args.device)

                pred = model(data, data_char).transpose(1, 2)
                loss = loss_fn(pred, target)
                dev_loss += loss.item()
                
                y_true.extend(dev_loader.dataset.idx2label_batch(target.tolist()))
                y_pred.extend(dev_loader.dataset.idx2label_batch(pred.argmax(dim=1).tolist()))
        
        dev_loss /= len(dev_loader.dataset)
        report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
        joint_acc, token_acc = accuracies(y_true, y_pred)
        print(f'Epoch {epoch+1:3d} | dev loss {dev_loss:.5f} | joint acc {joint_acc:.3f} | token acc {token_acc:.3f}')
        print(report)
        wandb.log({'dev_loss': dev_loss, 'joint_acc': joint_acc, 'token_acc': token_acc})

        # save model 
        torch.save(model.state_dict(), args.ckpt_dir.joinpath('model.pt'))
    
    # end of training session
    print(f"Training completed, model saved to '{args.ckpt_dir.joinpath('model.pt')}'.")


def parse_args() -> Namespace:
    dirs = parsers.parser_dirs('slot')
    base = parsers.parser_base()

    parser = ArgumentParser(parents=[dirs, base])
    parser.add_argument("--load_model", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_grad_norm", type=float, default=0)
    parser.add_argument("--num_epoch", type=int, default=30)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir = args.ckpt_dir.joinpath(args.model_id)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)