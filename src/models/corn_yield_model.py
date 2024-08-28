import os

import timm
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

from src.utils import get_extractor_in_features

__all__ = (
    "CornYieldCNNLSTM",
    "CornYieldCNN",
)


class CornYieldCNNLSTM(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float,
        bidirectional: bool = False,
        num_lstm_layers: int = 3,
    ):
        super().__init__()
        in_features = get_extractor_in_features(
            encoder_name,
            in_chans=in_channels,
        )

        self.backbone = nn.Sequential(
            timm.create_model(
                encoder_name,
                pretrained=False,
                num_classes=0,
                in_chans=in_channels,
            ),
            nn.Linear(in_features, hidden_size),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)

        fc_in_features = hidden_size * 2 + 3 + 5 if bidirectional else hidden_size + 3 + 5
        self.fc = nn.Linear(fc_in_features, output_size)

    def load_and_freeze_backbone(
        self,
        backbone_checkpoint_path: str,
        device=torch.device("cpu"),
    ):
        assert os.path.isfile(backbone_checkpoint_path)
        self.backbone.load_state_dict(
            torch.load(
                backbone_checkpoint_path,
                map_location=device,
            ),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        seq_data,
        valid_data_mask,
        nitrogen_level,
        location_id,
        norm_passing_days,
    ):
        batch_size, sequence_length, num_channels, height, width = seq_data.shape
        cnn_features = []
        lengths = []

        for i in range(batch_size):
            valid_indices = valid_data_mask[i] == 1
            valid_x = seq_data[i, valid_indices]

            cnn_features.append(
                torch.cat(
                    (
                        self.backbone(valid_x),
                        norm_passing_days[i, valid_indices][..., None],
                    ),
                    dim=1,
                )
            )
            lengths.append(valid_x.shape[0])

        cnn_features = rnn_utils.pad_sequence(cnn_features, batch_first=True)

        cnn_features = rnn_utils.pack_padded_sequence(
            cnn_features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, (hn, cn) = self.lstm(cnn_features)
        lstm_output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        idx = (torch.tensor(lengths) - 1).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, lstm_output.size(2))
        lstm_output = lstm_output.gather(1, idx.to(lstm_output.device)).squeeze(1)
        lstm_output = self.dropout(lstm_output)

        lstm_output = torch.cat(
            (
                lstm_output,
                nn.functional.one_hot(nitrogen_level, num_classes=3).float(),
                nn.functional.one_hot(location_id, num_classes=5).float(),
            ),
            axis=1,
        )

        output = self.fc(lstm_output)

        return output


class CornYieldCNN(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        hidden_size: int,
        output_size: int,
        pretrained: bool = True,
    ):
        super().__init__()
        in_features = get_extractor_in_features(
            encoder_name,
            in_chans=in_channels,
        )

        self.backbone = timm.create_model(
            encoder_name,
            pretrained=True,
            num_classes=0,
            in_chans=in_channels,
        )

        fc_in_features = in_features + 1 + 3 + 5
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def select_random_valid_samples(
        self,
        data,
        valid_mask,
        norm_passing_days,
    ):
        batch_size, seq_length, channel, height, width = data.shape

        valid_mask = valid_mask.bool()
        selected_indices = torch.zeros(batch_size, dtype=torch.long, device=data.device)

        for i in range(batch_size):
            valid_indices = torch.where(valid_mask[i])[0]

            if len(valid_indices) == 0:
                selected_indices[i] = torch.randint(0, seq_length, (1,))
            else:
                selected_indices[i] = valid_indices[torch.randint(0, len(valid_indices), (1,))]

        selected_samples = data[torch.arange(batch_size), selected_indices]
        norm_passing_days = norm_passing_days[torch.arange(batch_size), selected_indices]

        return selected_samples, norm_passing_days

    def forward_one_sample(
        self,
        seq_data,
        valid_data_mask,
        nitrogen_level,
        location_id,
        norm_passing_days,
    ):
        data, norm_passing_days = self.select_random_valid_samples(
            seq_data,
            valid_data_mask,
            norm_passing_days,
        )

        return self.forward(
            data,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

    def forward_sequence(
        self,
        seq_data,
        valid_data_mask,
        nitrogen_level,
        location_id,
        norm_passing_days,
    ):
        batch_size, seq_length, channel, height, width = seq_data.shape

        batch_output = []
        for i in range(batch_size):
            valid_indices = valid_data_mask[i] == 1
            valid_x = seq_data[i, valid_indices]
            valid_norm_passing_days = norm_passing_days[i, valid_indices]
            num_valids = valid_indices.sum().item()

            output = self.forward(
                valid_x,
                nitrogen_level[..., None][i].expand(num_valids),
                location_id[..., None][i].expand(num_valids),
                valid_norm_passing_days,
            )

            valid_norm_passing_days = torch.exp(valid_norm_passing_days)
            output = output.squeeze(1) * valid_norm_passing_days / torch.sum(valid_norm_passing_days)
            output = output.sum(dim=0)[..., None]
            batch_output.append(output)

        batch_output = torch.cat(
            batch_output,
            dim=0,
        )[..., None]

        return batch_output

    def forward(
        self,
        data,
        nitrogen_level,
        location_id,
        norm_passing_days,
    ):
        output = self.backbone(
            data,
        )
        output = torch.cat(
            (
                output,
                norm_passing_days[..., None],
                nn.functional.one_hot(nitrogen_level, num_classes=3).float(),
                nn.functional.one_hot(location_id, num_classes=5).float(),
            ),
            dim=1,
        )

        output = self.fc(output)

        return output
