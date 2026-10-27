"""QuEEN-DTA network: multi-branch encoder with a residual block and squeeze-excitation gate."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from queen_dta.config import QUEEN_DROPOUT
from queen_dta.features import QUEEN_VOCAB_SIZE


def keras_init(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.05, 0.05)


class QueenNet(nn.Module):
    def __init__(self, smi_dim, bp_dim, ngram_dim, dropout=QUEEN_DROPOUT):
        super().__init__()
        self.drug_fc1 = nn.Linear(smi_dim, 1024)
        self.drug_bn = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.01)
        self.drug_fc2 = nn.Linear(1024, 512)
        self.embedding = nn.Embedding(QUEEN_VOCAB_SIZE, 40)
        self.conv1 = nn.Conv1d(40, 64, 5)
        self.conv2 = nn.Conv1d(64, 128, 5)
        self.bp_fc = nn.Linear(bp_dim, 32)
        self.ngram_fc = nn.Linear(ngram_dim, 128)
        self.res_fc1 = nn.Linear(512 + 128 + 32 + 128, 512)
        self.res_bn = nn.BatchNorm1d(512, eps=1e-3, momentum=0.01)
        self.res_fc2 = nn.Linear(512, 512)
        self.res_shortcut = nn.Linear(512 + 128 + 32 + 128, 512)
        self.se_fc1 = nn.Linear(512, 256)
        self.se_fc2 = nn.Linear(256, 512)
        self.fc = nn.Linear(512, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.apply(keras_init)

    def forward(self, x_smi, x_prot, x_bp, x_ngram):
        d = F.relu(self.drug_fc1(x_smi))
        d = self.drug_bn(d)
        d = self.dropout(d)
        d = F.relu(self.drug_fc2(d))

        p = self.embedding(x_prot.long()).transpose(1, 2)
        p = F.relu(self.conv1(p))
        p = F.relu(self.conv2(p))
        p = p.amax(dim=2)

        b = F.relu(self.bp_fc(x_bp))
        g = F.relu(self.ngram_fc(x_ngram))

        z = torch.cat([d, p, b, g], dim=1)
        r = F.relu(self.res_fc1(z))
        r = self.res_bn(r)
        r = F.relu(self.res_fc2(r))
        z = r + self.res_shortcut(z)

        se = F.relu(self.se_fc1(z))
        se = torch.sigmoid(self.se_fc2(se))
        z = z * se

        z = F.relu(self.fc(z))
        z = self.dropout(z)
        return self.out(z)


def queen_forward(model, tensors, rows):
    return model(*[t[rows] for t in tensors])
