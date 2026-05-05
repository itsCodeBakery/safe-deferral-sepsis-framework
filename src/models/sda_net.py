
import torch
import torch.nn as nn


class SDANet(nn.Module):
    """
    SDA-Net: Safe Deferral Agent Network for Early Sepsis Prediction.

    Input:
        x: Tensor of shape (batch_size, sequence_length, num_features)
           containing normalized clinical values.
        mask: Tensor of shape (batch_size, sequence_length, num_features)
              where 1 indicates observed values and 0 indicates missing values.

    Output:
        risk_prob: Probability of sepsis.
        action_logits: Logits for three actions: wait, alert, defer.
        action_probs: Softmax probabilities for wait, alert, defer.
    """

    def __init__(
        self,
        num_features=40,
        hidden_dim=128,
        num_layers=2,
        dropout=0.30
    ):
        super(SDANet, self).__init__()

        self.num_features = num_features
        self.input_dim = num_features * 2
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.risk_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Linear(hidden_dim, 3)

    def forward(self, x, mask):
        z = torch.cat([x, mask], dim=-1)

        gru_out, _ = self.gru(z)

        # Last time-step representation
        h_last = gru_out[:, -1, :]

        h = self.shared(h_last)

        risk_logit = self.risk_head(h).squeeze(-1)
        risk_prob = torch.sigmoid(risk_logit)

        action_logits = self.action_head(h)
        action_probs = torch.softmax(action_logits, dim=-1)

        return risk_logit, risk_prob, action_logits, action_probs
