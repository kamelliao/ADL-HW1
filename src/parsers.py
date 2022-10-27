from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch


def parser_dirs(task: str) -> ArgumentParser:
    """_summary_

    Args:
        task (str): 'intent' | 'slot'

    Returns:
        ArgumentParser: _description_
    """
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default=f"./data/{task}/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default=f"./cache/{task}/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default=f"./ckpt/{task}/",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=str(datetime.now())
    )

    return parser


def parser_base() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)

    # random seed
    parser.add_argument('--seed', type=int, default=1001) 

    # data
    parser.add_argument("--max_len", type=int, default=25)

    # model
    parser.add_argument("--model", type=str, default='LSTM',
                        help='type of network (RNN_RELU, RNN_TANH, LSTM, GRU, CNN)')
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    parser.add_argument("--embed_type", type=str, default='sum', help='(last, sum, mean, learnt)')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )

    return parser