import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    """
    BiLSTM-based text classifier with embedding projection and dropout.
    """
    def __init__(
        self,
        bert_embedding_dim: int = 768,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 6,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
        fc_dropout: float = 0.3,
        input_dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.embedding_projection = nn.Linear(bert_embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            embeddings (Tensor): Input tensor of shape [batch_size, seq_len, bert_embedding_dim].

        Returns:
            Tensor: Output logits of shape [batch_size, output_dim].
        """
        x = self.embedding_projection(embeddings)
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        lstm_output, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]

        final_hidden = self.fc_dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits


class MyGRU(nn.Module):
    """
    BiGRU-based text classifier with embedding projection, normalization, and dropout.
    """
    def __init__(
        self,
        bert_embedding_dim: int = 768,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 6,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3,
        fc_dropout: float = 0.1,
        input_dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.embedding_projection = nn.Linear(bert_embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (Tensor): Input tensor of shape [batch_size, seq_len, bert_embedding_dim].

        Returns:
            Tensor: Output logits of shape [batch_size, output_dim].
        """
        x = self.embedding_projection(embeddings)
        x = self.norm(x)
        x = self.input_dropout(x)

        _, h_n = self.gru(x)

        if self.gru.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]

        final_hidden = self.fc_dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits


# C-LSTM: https://arxiv.org/pdf/1511.08630

class HybridNN(nn.Module):
    """
    Hybrid CNN + BiLSTM + Attention-based text classifier.
    Applies 1D convolution followed by LSTM and attention pooling.

    Args:
        embedding_dim (int): Dimensionality of input embeddings.
        conv_out_channels (int): Number of channels from the CNN layer.
        kernel_size (int): Kernel size for the Conv1D layer.
        hidden_dim (int): LSTM hidden size.
        output_dim (int): Number of output classes.
        lstm_layers (int): Number of LSTM layers.
        bidirectional (bool): Use bidirectional LSTM.
        dropout (float): Dropout for LSTM.
        fc_dropout (float): Dropout before FC layer.
        input_dropout (float): Dropout on input embeddings.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        conv_out_channels: int = 64,
        kernel_size: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 6,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.4,
        fc_dropout: float = 0.2,
        input_dropout: float = 0.3
    ) -> None:
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)

        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=1
        )
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=32)
        self.relu = nn.ReLU()
        self.spatial_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )

        self.attn_linear = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

        self.fc_dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (Tensor): Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Tensor: Output logits [batch_size, output_dim]
        """
        # Input dropout
        x = self.input_dropout(embeddings)

        # CNN: [B, seq_len, emb_dim] -> [B, emb_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # return to [B, seq_len, conv_out_channels]

        x = self.spatial_dropout(x)

        # LSTM part
        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_dim * num_directions]

        # Attention part
        attn_weights = torch.softmax(self.attn_linear(lstm_out), dim=1)  # [B, seq_len, 1]
        attn_output = torch.sum(lstm_out * attn_weights, dim=1)          # [B, hidden_dim * num_directions]

        # Final classification
        x = self.fc_dropout(attn_output)
        logits = self.classifier(x)
        return logits