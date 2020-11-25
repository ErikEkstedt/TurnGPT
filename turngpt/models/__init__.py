import torch.nn as nn


class KQV(nn.Module):
    def __init__(self, D=32, features_in=14, features_out=1):
        super().__init__()
        self.KeyValue = nn.Linear(D, D * 2)
        self.Q = nn.Linear(
            D * features_in, D * features_out
        )  # condense total representation for a single query
        # self.key_query_val = nn.Linear(D, D * 3)

    def forward(self, x):
        """
        x: (B, T, features, D)
        k: (B, T, features, D)  # all features in
        v: (B, T, features, D)  # all features in
        q: (B, T, 1, D)  # single query out
        """
        # k, q, v = self.key_query_val(x).chunk(3, dim=-1)
        k, v = self.KeyValue(x).chunk(2, dim=-1)
        q = self.Q(x.flatten(-2)).unsqueeze(-2)  # unsqueze feature dim
        return k, q, v


class Attention1D(nn.Module):
    def __init__(self, D, features, features_out, num_heads):
        super().__init__()
        self.D = D
        self.features = features
        self.features_out = features_out

        self.key_query_val = KQV(D, features_in=features, features_out=features_out)
        self.attn = nn.MultiheadAttention(D, num_heads=num_heads)

    def forward(self, x):
        """
        Expects x: (B, T, n_feats, D)
        """
        B, N, n_feats, D = x.size()
        k, q, v = self.key_query_val(x)

        # condense batch and time dim:
        # (B, T, features, D) -> (B*T, features, D)
        # (B*T, features, D) -> (features, B*T, D)
        # permute for multihead attention
        k = k.view(-1, self.features, self.D).permute(1, 0, 2)
        v = v.view(-1, self.features, self.D).permute(1, 0, 2)
        q = q.view(-1, self.features_out, self.D).permute(1, 0, 2)
        # print("k: ", tuple(k.shape))  # (features, BT, D)
        # print("v: ", tuple(v.shape))  # (features, BT, D)
        # print("q: ", tuple(q.shape))  # (features_out, BT, D)

        z, att = self.attn(q, k, v)
        # print("z: ", tuple(z.shape))  # features_out, BT, D
        # print("att: ", tuple(att.shape))  # BT, features_out, features_in
        # print("-" * 30)

        # reshape back
        # (features, BT, D) -> (BT, features, D) -> (B, T, features, D)
        z = z.permute(1, 0, 2).view(
            B, -1, self.features_out, self.D
        )  # B, T, feature, D
        att = att.view(B, -1, self.features_out, self.features)  # B, T, feature, D
        return z, att
