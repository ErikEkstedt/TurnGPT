import torch
from torch.nn.utils.rnn import pad_sequence

from turngpt.tokenizer import SpokenDialogTokenizer
from turngpt.models.pretrained import TurnGPTModel
from ttd.utils import read_json
import re



if __name__ == "__main__":
    # get input tokens

    checkpoint = "/home/erik/projects/checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981_V2.ckpt"
    tokenizer = SpokenDialogTokenizer(pretrained_tokenizer="gpt2")
    PAD = tokenizer._tokenizer.pad_token_id
    model = TurnGPTModel.load_from_checkpoint(checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    dialog_list = read_json('/home/erik/ExperimentReal/dialogs_text.json')
    input_ids, speaker_ids, lens = [], [], []
    for turn_list in dialog_list:
        turn_list = [re.sub('^\s', '', turn) for turn in turn_list]
        inp, sp= tokenizer.turns_to_turngpt_tensors(
            turn_list, add_end_speaker_token=False
        )
        input_ids.append(inp.squeeze())
        speaker_ids.append(sp.squeeze())
        lens.append(inp.nelement())
        assert inp.nelement() == sp.nelement()
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD)
    speaker_ids = pad_sequence(speaker_ids, batch_first=True, padding_value=PAD)

    labels = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    speaker_ids = speaker_ids[:, :-1].contiguous()
    labels[labels == PAD] = -100  # don't train on these

    with torch.no_grad():
        logits = model(
                input_ids.to(model.device), 
                speaker_ids.to(model.device)
                )['logits'].cpu()

    ce = torch.nn.CrossEntropyLoss()(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    ppl = torch.exp(ce)
    print('ce: ', ce.item())
    print('perplexity: ', ppl.item())
