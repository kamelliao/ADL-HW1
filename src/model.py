from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        model: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        embed_type: str,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        self.hidden_size = hidden_size
        self.d = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.num_class = num_class
        self.embed_type = embed_type
        self.model = model

        # TODO: model architecture
        if model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, model)(
                300, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
        else:
            nonlinearity = 'relu' if model == 'RNN_RELU' else 'tanh'
            self.rnn = nn.RNN(
                300, hidden_size, num_layers,
                nonlinearity=nonlinearity, batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
        self.fc = nn.Linear(self.encoder_output_size, num_class)
        self.aggregate = nn.Conv1d(25, 1, 1)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.d * self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed(batch)
        output, _ = self.rnn(batch)
        
        if self.embed_type == 'sum':
            embed = output.sum(dim=1)
        elif self.embed_type == 'mean':
            embed = output.mean(dim=1)
        elif self.embed_type == 'learnt':
            embed = self.aggregate(output).squeeze(1)
        else:
            embed = output[:, -1, :]  # use the embbeding of the last token

        logits = self.fc(embed)
        return logits

    def init_hidden_state(self, batch_size):
        return torch.randn(self.d*self.num_layers, batch_size, self.hidden_size, device='cuda')


class SeqTagger(SeqClassifier):
    def __init__(self, char_embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.char_embed = Embedding.from_pretrained(char_embedding, freeze=False)
        if self.model == 'CNN-biLSTM':
            self.rnn = nn.LSTM(
                315, self.hidden_size, self.num_layers,
                batch_first=True, dropout=args[4], bidirectional=args[5]
            )
        self.cnn = nn.Sequential(
            nn.Conv1d(25, 15, 2),
            nn.AdaptiveMaxPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.num_class),
        )

    def forward(self, batch, batch_char) -> Dict[str, torch.Tensor]:
        batch_token = self.embed(batch)
        if self.model == 'CNN-biLSTM':
            batch_features = torch.stack([
                torch.cat([batch_t, self.cnn(self.char_embed(batch_c).permute(0, 2, 1)).squeeze(-1)], dim=-1) for batch_t, batch_c in zip(batch_token, batch_char)
            ])
        else:
            batch_features = batch_token

        output, _ = self.rnn(batch_features)

        logits = self.fc(output)
        return logits


class CNNClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        dropout: float,
        num_class: int,
        kernels: List[int] = [2, 3, 4],
        couts: List[int] = [100, 100, 100],
    ):
        super(CNNClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        self.cnns = nn.ModuleList([self.convLayer(kernel, cout) for (kernel, cout) in zip(kernels, couts)])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(sum(couts), num_class)
        )

    def forward(self, x):
        embed = self.embed(x).permute(0, 2, 1)  # output=(batch, channel(embedding_size), sequence_len)
        outputs = [conv(embed) for conv in self.cnns]
        output = torch.cat(outputs, dim=1).squeeze(dim=-1)
        logits = self.fc(output)

        return logits
    
    @staticmethod
    def convLayer(kernel, cout, cin=300):
        return nn.Sequential(
            nn.Conv1d(cin, cout, kernel),
            nn.AdaptiveMaxPool1d(1),
            nn.ReLU()
        )