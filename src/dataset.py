from typing import List, Dict

from torch import LongTensor
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        split: str,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.split = split
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text = [sample['text'].split(' ') for sample in samples]
        text_ids = self.vocab.encode_batch(text, self.max_len)
        
        if self.split == 'test':
            doc_ids = [sample['id'] for sample in samples] 
            return LongTensor(text_ids), doc_ids
 
        label = [self.label_mapping[sample['intent']] for sample in samples]
        return LongTensor(text_ids), LongTensor(label)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100
    def __init__(self, char_vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_vocab = char_vocab

    def collate_fn(self, samples):
        tokens = [sample['tokens'] for sample in samples]
        tokens_ids = self.vocab.encode_batch(tokens, self.max_len)
        tokens_char_ids = [self.char_vocab.encode_batch(token, self.max_len) for token in pad_to_len(tokens, self.max_len, Vocab.PAD)]
        # breakpoint()

        if self.split == 'test':
            doc_ids = [sample['id'] for sample in samples]
            original_len = [len(sample['tokens']) for sample in samples]
            return LongTensor(tokens_ids), LongTensor(tokens_char_ids), doc_ids, original_len

        tags = [[self.label2idx(tag) for tag in sample['tags']] for sample in samples]
        tags_padded = pad_to_len(tags, to_len=self.max_len, padding=8)
        return LongTensor(tokens_ids), LongTensor(tokens_char_ids), LongTensor(tags_padded)

    def idx2label_batch(self, batch_ids: List[List[int]]) -> List[List[str]]:
        labels = [[self.idx2label(idx) for idx in ids] for ids in batch_ids]
        return labels