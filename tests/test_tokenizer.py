import pytest
import torch
from turngpt.tokenizer import SpokenDialogTokenizer


INPUT = {
    "list_a": [
        "Hello there I basically had the worst day of my life",
        "Oh no, what happened?",
        "Do you want the long or the short story?",
    ],
    "list_b": [
        "Hello there I basically had the worst day of my life",
        "Oh no, what happened?",
    ],
}
INPUT["list_of_list"] = [INPUT["list_a"], INPUT["list_b"]]


string_a = 'Yesterday Hello ther, "honey"<ts> godday... you are great<ts> Not as good as you!<ts>'
string_b = 'Yesterday hello ther , "honey"<ts> godday... you are great<ts> Not as good as you!<ts>'


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("gpt2", True)),
        pytest.param(("gpt2", False)),
        # pytest.param("microsoft/DialoGPT-small", marks=pytest.mark.slow),
        # pytest.param("microsoft/DialoGPT-medium", marks=pytest.mark.slow),
        # pytest.param("microsoft/DialoGPT-large", marks=pytest.mark.slow),
    ],
)
def tokenizer(request):
    name_or_path = request.param[0]
    normalization = request.param[1]
    return SpokenDialogTokenizer(
        pretrained_model_name_or_path=name_or_path,
        normalization=normalization,
    )


@pytest.mark.tokenizer
def test_eos_token(tokenizer):
    assert tokenizer.eos_token == "<ts>"


@pytest.mark.tokenizer
def test_speaker_tokens(tokenizer):
    assert tokenizer.sp1_token == "<speaker1>"
    assert tokenizer.sp2_token == "<speaker2>"


@pytest.mark.tokenizer
def test_post_punctuation_space(tokenizer):
    s = "hello,,,there;everybody.whats up<ts> how are you<ts>"
    t = tokenizer(s)

    if tokenizer.normalization:
        ans = "hello there everybody whats up<ts> how are you<ts>"
    else:
        ans = s

    out = tokenizer.decode(t["input_ids"])
    assert out == ans


@pytest.mark.tokenizer
def test_double_spaces(tokenizer):
    s = "hello there how are you doin<ts>     how are you<ts>"
    t = tokenizer(s)

    if tokenizer.normalization:
        ans = "hello there how are you doin<ts> how are you<ts>"
    else:
        ans = s

    out = tokenizer.decode(t["input_ids"])
    assert out == ans


@pytest.mark.tokenizer
def test_speaker_ids(tokenizer):
    s = "Hello there, how are you today?<ts> I'm doing good thank you!<ts> That's great<ts>"
    outputs = tokenizer(s)
    out_string = tokenizer.decode(outputs["input_ids"])

    if tokenizer.normalization:
        correct_string = "hello there how are you today<ts> i'm doing good thank you<ts> that's great<ts>"
        correct_speaker = [tokenizer.sp1_token_id] * 7
        correct_speaker += [tokenizer.sp2_token_id] * 7
        correct_speaker += [tokenizer.sp1_token_id] * 4
    else:
        correct_string = "Hello there, how are you today?<ts> I'm doing good thank you!<ts> That's great<ts>"
        correct_speaker = [tokenizer.sp1_token_id] * 9
        correct_speaker += [tokenizer.sp2_token_id] * 8
        correct_speaker += [tokenizer.sp1_token_id] * 4

    assert out_string == correct_string
    assert outputs["speaker_ids"] == correct_speaker

    out = tokenizer([["hello", "bye"], ["hello", "bye", "you"]], include_end_ts=False)
    assert type(out["input_ids"]) == type(out["speaker_ids"])
    assert len(out["input_ids"]) == len(out["speaker_ids"])

    out = tokenizer(["hello", "bye"], include_end_ts=False)
    assert type(out["input_ids"]) == type(out["speaker_ids"])
    assert len(out["input_ids"]) == len(out["speaker_ids"])

    out = tokenizer("hello", include_end_ts=False)
    assert type(out["input_ids"]) == type(out["speaker_ids"])
    assert len(out["input_ids"]) == len(out["speaker_ids"])


@pytest.mark.tokenizer
def test_idx_to_tokens(tokenizer):
    turn_list = [
        "hello there how are you doing today?",
        "I'm doing very well thank you, how about you?",
        "well, I'm sad",
    ]
    tok_out = tokenizer(turn_list, include_end_ts=False)
    ids_list = tok_out["input_ids"]
    ids_tens = torch.tensor(tok_out["input_ids"])
    t1 = tokenizer.idx_to_tokens(ids_list)
    t2 = tokenizer.idx_to_tokens(ids_tens)
    t3 = tokenizer.idx_to_tokens(ids_list[0])
    assert t1 == t2
    assert isinstance(t3, str)


@pytest.mark.tokenizer
def test_string_tokenization(tokenizer):
    string_a = 'Yesterday Hello ther, "honey"<ts> godday... you are great<ts> Not as good as you!<ts>'

    if tokenizer.normalization:
        ans = "yesterday hello ther honey<ts> godday you are great<ts> not as good as you<ts>"
    else:
        ans = string_a

    outputs = tokenizer(string_a)
    assert tokenizer.decode(outputs["input_ids"]) == ans


@pytest.mark.tokenizer
def test_list_tokenization(tokenizer):
    text_list = [
        'Yesterday Hello ther, "honey"',
        "godday... you are great",
        "Not as good as you!",
    ]

    if tokenizer.normalization:
        ans = "yesterday hello ther honey<ts> godday you are great<ts> not as good as you<ts>"
    else:
        ans = "<ts> ".join(text_list) + "<ts>"

    outputs = tokenizer(text_list)
    assert tokenizer.decode(outputs["input_ids"]) == ans


@pytest.mark.tokenizer
def test_list_of_lists_tokenization(tokenizer):
    text_list = [
        'Yesterday Hello ther, "honey"',
        "godday... you are great",
        "Not as good as you!",
    ]
    list_of_lists = [text_list, text_list[:-1], text_list[:-2]]

    if tokenizer.normalization:
        output_list_of_lists = [
            "yesterday hello ther honey<ts> godday you are great<ts> not as good as you<ts>",
            "yesterday hello ther honey<ts> godday you are great<ts>",
            "yesterday hello ther honey<ts>",
        ]
    else:
        output_list_of_lists = [
            'Yesterday Hello ther, "honey"<ts> godday... you are great<ts> Not as good as you!<ts>',
            'Yesterday Hello ther, "honey"<ts> godday... you are great<ts>',
            'Yesterday Hello ther, "honey"<ts>',
        ]
    outputs = tokenizer(list_of_lists)

    output_strings = []
    for out in outputs["input_ids"]:
        output_strings.append(tokenizer.decode(out))

    assert output_strings[0] == output_list_of_lists[0]
    assert output_strings[1] == output_list_of_lists[1]
    assert output_strings[2] == output_list_of_lists[2]
