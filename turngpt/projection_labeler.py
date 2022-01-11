import torch
import torch.nn as nn


# TODO: tests
class ProjectionLabeler(nn.Module):
    def __init__(self, projection_steps, token_id):
        super().__init__()
        self.projection_steps = projection_steps
        self.token_id = token_id

        self.labeler = nn.ConvTranspose1d(
            in_channels=1, out_channels=1, kernel_size=projection_steps, bias=False
        )
        self.labeler.weight.data.fill_(1.0)
        self.labeler.requires_grad_(False)
        self.offset = projection_steps - 1

    @torch.no_grad()
    def forward(self, input_ids):

        # prepare inputs
        eos = (input_ids == self.token_id) * 1.0
        eos = eos.unsqueeze(1)

        # construct labels
        proj_label = self.labeler(eos)

        # offset the transpose-output and remove channel
        proj_label = proj_label[..., self.offset :].squeeze(1)
        return proj_label
