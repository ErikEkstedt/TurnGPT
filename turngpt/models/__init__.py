import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_check_batch(inp, model, n_batch=1):
    opt = torch.optim.SGD(model.parameters(), lr=1)
    opt.zero_grad()
    inp.requires_grad = True
    if inp.grad is not None:
        inp.grad.zero_()

    # forward
    z = model(inp.to(model.device))
    z[n_batch].sum().backward()

    # calc grad
    x_grad = inp.grad.data.abs()
    for i in range(inp.ndim - 1):
        x_grad = x_grad.sum(dim=-1)

    if x_grad.sum() == x_grad[n_batch]:
        print("Gradient Batch Success!")
    else:
        print("Gradient Batch Failure!")


def gradient_check_word_time(inp, model):
    opt = torch.optim.SGD(model.parameters(), lr=1)
    opt.zero_grad()
    inp.requires_grad = True
    if inp.grad is not None:
        inp.grad.zero_()
    N = inp.shape[1]
    target = N // 2
    z = model(inp.to(model.device))
    z[:, target].sum().backward()
    x_grad = inp.grad.data.abs().sum(dim=0)  # sum batches
    x_grad = x_grad.sum(dim=-1)  # sum prosody feats
    if inp.ndim == 4:
        x_grad = x_grad.sum(dim=-1)  # sum T
    if x_grad[target + 1 :].sum() == 0:
        print("Gradient Step Success!")
    else:
        print("Gradient Step Failure!")


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


class SelfAttention(nn.Module):
    """
    Source:
        https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(self, hidden, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert hidden % n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(hidden, hidden)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_frames=100,
        hidden=32,
        in_channels=1,
        num_layers=7,
        kernel_size=3,
        strides=2,
        padding=0,
        activation="ReLU",
        independent=False,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.in_channels = in_channels
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = getattr(nn, activation)

        if isinstance(self.strides, int):
            self.strides = [1] * self.num_layers
            self.strides[0] = strides
        else:
            assert len(self.strides) == num_layers, "provide as many strides as layers"

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * self.num_layers
        else:
            assert (
                len(kernel_size) == num_layers
            ), "provide as many kernel_size as layers"

        if isinstance(padding, int):
            self.padding = [padding] * self.num_layers
        else:
            assert len(padding) == num_layers, "provide as many padding as layers"

        if independent:
            self.groups = in_channels
            self.hidden = hidden
        else:
            self.groups = 1
            self.hidden = hidden

        self.convs = self._build_conv_layers()
        self.output_size = self.calc_conv_out()

    def conv_output_size(self, input_size, kernel_size, padding, stride):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def calc_conv_out(self):
        n = self.input_frames
        for i in range(0, self.num_layers):
            n = self.conv_output_size(
                n, self.kernel_size[i], self.padding[i], stride=self.strides[i]
            )
        return n

    def _build_conv_layers(self):
        layers = [
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.hidden,
                kernel_size=self.kernel_size[0],
                stride=self.strides[0],
                padding=self.padding[0],
                groups=self.groups,
            ),
            nn.BatchNorm1d(self.hidden),
            self.activation(),
        ]
        for i in range(1, self.num_layers):
            layers += [
                nn.Conv1d(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    kernel_size=self.kernel_size[i],
                    stride=self.strides[i],
                    padding=self.padding[i],
                    groups=self.groups,
                ),
                nn.BatchNorm1d(self.hidden),
                self.activation(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-2)  # B, T -> B, 1, T
        x = self.convs(x)
        return x


class VectorQuantizerEMA(nn.Module):
    """
    Slightly changed version of Zalando research vq-vae implementation
    Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def perplexity(self, encodings):
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC

        # Flatten input
        B, N, D = inputs.shape
        input_shape = inputs.shape
        flat_input = inputs.contiguous().view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # encodings
        encodings = encodings.view(B, N, self._num_embeddings)
        return quantized, encodings, loss
