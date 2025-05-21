import torch.nn as nn
import torch

class MyLSTM(nn.Module):
    def __init__(self, bert_embedding_dim=768,embedding_dim=128, hidden_dim=64, output_dim=6, num_layers=1, bidirectional=False, dropout=0.3, fc_dropout=0.3, input_dropout=0.2):
        super(MyLSTM, self).__init__()

        self.embedding_projection = nn.Linear(bert_embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout) # 

        self.lstm = nn.LSTM(
            input_size=embedding_dim, # 768
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        # LAYER 2: Fully-connected
        self.fc_dropout = nn.Dropout(fc_dropout)
        # self.fc = nn.Linear(
        #     hidden_dim * (2 if bidirectional else 1),
        #     output_dim
        # )
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        )

    def forward(self, bert_embeddings):  # [batch_size, seq_len, embedding_dim]

        # Dropout on BERT-embeddings
        x = self.embedding_projection(bert_embeddings)
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        lstm_output, (h_n, c_n) = self.lstm(x)

        if self.lstm.bidirectional:
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]

        h_final = self.fc_dropout(h_final)
        out = self.fc(h_final)
        return out
    
class MyGRU(nn.Module):
    def __init__(self, bert_embedding_dim=768, embedding_dim=256, hidden_dim=128, output_dim=6, num_layers=2, bidirectional=False, dropout=0.3, fc_dropout=0.1, input_dropout=0.2):
        super(MyGRU, self).__init__()

        # reduce bert embeds to embedding_dim
        self.embedding_projection = nn.Linear(bert_embedding_dim, embedding_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.norm = nn.LayerNorm(embedding_dim)

        # GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
         # [batch_size, seq_len, embedding_dim]
        x = self.embedding_projection(x)
        x = self.norm(x) 
        x = self.input_dropout(x)
        # Pass through GRU
        gru_output, h_n = self.gru(x)

        # Extract final hidden state
        if self.gru.bidirectional:
            h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_final = h_n[-1]

        # Apply dropout and FC layer
        h_final = self.fc_dropout(h_final)
        out = self.fc(h_final)
        return out
    
# C-LSTM: https://arxiv.org/pdf/1511.08630

class HybridNN(nn.Module):
    def __init__(self, embedding_dim=768, conv_out_channels=64, kernel_size=5, 
                 hidden_dim=64, output_dim=6, lstm_layers=1, bidirectional=True, 
                 dropout=0.4, fc_dropout=0.2, input_dropout=0.3):
        super(HybridNN, self).__init__()

        self.input_dropout = nn.Dropout(input_dropout)

        # CNN block
        self.conv1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=conv_out_channels,
                               kernel_size=kernel_size,
                               padding=1)
        # self.conv2 = nn.Conv1d(in_channels=conv_out_channels,
        #                        out_channels=conv_out_channels,
        #                        kernel_size=kernel_size,
        #                        padding=1)

        self.spatial_dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(input_size=conv_out_channels,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if lstm_layers > 1 else 0,
                            batch_first=True)

        self.attn_linear = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

        # Output
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

        self.max1d = nn.AdaptiveMaxPool1d(output_size=32)

    def forward(self, embeddings):
        # Dropout input
        x = self.input_dropout(embeddings)  # [batch_size, seq_len, embedding_dim]
        # Conv1D
        x = x.permute(0, 2, 1)                 # [B, seq, emb]
        x = self.relu(self.max1d(self.conv1(x)))
        # x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)                 # LSTM
        x = self.spatial_dropout(x) 


        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]

        #attention mechanism
        attn_weights = torch.softmax(self.attn_linear(lstm_out), dim=1)  # [batch_size, seq_len, 1]
        attn_output = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_dim*2]

        # FC
        out = self.fc_dropout(attn_output)
        out = self.fc(out)

        return out