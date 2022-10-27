from tkinter import W
from typing import Iterable, List, Tuple

from torch import FloatTensor


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }
    
    def __len__(self):
        return len(self.token2idx)

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds


def gradient_norm(model):
    param_norms = [param.grad.detach().data.norm(2) for param in model.parameters()]
    total_norm = FloatTensor(param_norms).norm(2).item()
    return total_norm


def accuracies(pred: List[List[str]], true: List[List[str]]) -> Tuple[float, float]:
    assert len(pred) == len(true)

    correct = 0
    token_total = 0
    token_correct = 0
    for p, t in zip(pred, true):
        # joint
        correct += (p == t)

        # token
        token_total += len(p)
        for p_token, t_token in zip(p, t):
            token_correct += (p_token == t_token)
    
    joint_accuracy = correct / len(pred)
    token_accuracy = token_correct / token_total

    return joint_accuracy, token_accuracy