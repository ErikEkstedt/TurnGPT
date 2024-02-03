import pytest
import torch

from turngpt.tokenizer import SpokenDialogTokenizer


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("gpt2", True)),
        # pytest.param(("gpt2", False)),
        # pytest.param("microsoft/DialoGPT-small", marks=pytest.mark.slow),
        # pytest.param("microsoft/DialoGPT-medium", marks=pytest.mark.slow),
        # pytest.param("microsoft/DialoGPT-large", marks=pytest.mark.slow),
    ],
)
def tokenizer(request) -> SpokenDialogTokenizer:
    name_or_path = request.param[0]
    normalization = request.param[1]
    return SpokenDialogTokenizer(
        pretrained_model_name_or_path=name_or_path,
        normalization=normalization,
    )


@pytest.mark.tokenizer
def test_tokens(tokenizer: SpokenDialogTokenizer) -> None:
    assert tokenizer.eos_token == "<ts>", "Wrong eos token"
    assert tokenizer.sp1_token == "<speaker1>", "Wrong sp1_token token"
    assert tokenizer.sp2_token == "<speaker2>", "Wrong sp2_token token"


@pytest.mark.tokenizer
def test_post_punctuation_space(tokenizer: SpokenDialogTokenizer):
    s = "hello,,,there;everybody.whats up<ts> how are you<ts>"
    t = tokenizer(s)

    if tokenizer.normalization:
        ans = "hello there everybody whats up<ts> how are you<ts>"
    else:
        ans = s

    out = tokenizer.decode(t["input_ids"])
    assert out == ans


@pytest.mark.tokenizer
def test_double_spaces(tokenizer: SpokenDialogTokenizer):
    s = "hello there how are you doin<ts>     how are you<ts>"
    t = tokenizer(s)

    if tokenizer.normalization:
        ans = "hello there how are you doin<ts> how are you<ts>"
    else:
        ans = s

    out = tokenizer.decode(t["input_ids"])
    assert out == ans


@pytest.mark.tokenizer
def test_speaker_ids(tokenizer: SpokenDialogTokenizer):
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
def test_string_tokenization(tokenizer: SpokenDialogTokenizer):
    text_string = "Hello there how are you today?"
    ans_string = (
        "hello there how are you today" if tokenizer.normalization else text_string
    )
    ans_input_ids = [31373, 612, 703, 389, 345, 1909]

    t = tokenizer(
        text_string, include_end_ts=False, include_pre_space=False, return_tensors=None
    )
    ts_pred = tokenizer.decode(t["input_ids"])

    assert ts_pred == ans_string, f"Expected {ans_string} but got {ts_pred}"
    assert (
        t["input_ids"] == ans_input_ids
    ), f"Expected {ans_input_ids} but got {t['input_ids']}"

    # Test with return_tensors
    t = tokenizer(
        text_string, include_end_ts=False, include_pre_space=False, return_tensors="pt"
    )
    ts_pred = tokenizer.decode(t["input_ids"][0])
    assert ts_pred == ans_string, f"Expected {ans_string} but got {ts_pred}"
    assert (
        t["input_ids"] == torch.tensor(ans_input_ids)
    ).all(), "Expected the same tensor"


@pytest.mark.tokenizer
def test_string_tokenization_includes(tokenizer: SpokenDialogTokenizer):
    text_string = "Hello there how are you today?"
    ans_string = (
        "hello there how are you today" if tokenizer.normalization else text_string
    )
    ans_input_ids = [31373, 612, 703, 389, 345, 1909]
    ans_inp_pre_space = [23748, 612, 703, 389, 345, 1909]

    # With `include_end_ts`
    t = tokenizer(
        text_string, include_end_ts=True, include_pre_space=False, return_tensors=None
    )
    ts_pred = tokenizer.decode(t["input_ids"])
    assert (
        ts_pred == ans_string + tokenizer.eos_token
    ), f"Expected {ans_string} but got {ts_pred}"
    assert t["input_ids"] == ans_input_ids + [
        tokenizer.eos_token_id
    ], f"Expected {ans_input_ids} but got {t['input_ids']}"

    # With `include_pre_space`
    t = tokenizer(
        text_string, include_end_ts=False, include_pre_space=True, return_tensors=None
    )
    ts_pred = tokenizer.decode(t["input_ids"])
    assert ts_pred == " " + ans_string, f"Expected {ans_string} but got {ts_pred}"
    assert (
        t["input_ids"] == ans_inp_pre_space
    ), f"Expected {ans_input_ids} but got {t['input_ids']}"

    # Test with both
    t = tokenizer(
        text_string, include_end_ts=True, include_pre_space=True, return_tensors=None
    )
    ts_pred = tokenizer.decode(t["input_ids"])
    assert (
        ts_pred == " " + ans_string + tokenizer.eos_token
    ), f"Expected {ans_string} but got {ts_pred}"
    assert t["input_ids"] == ans_inp_pre_space + [
        tokenizer.eos_token_id
    ], f"Expected {ans_input_ids} but got {t['input_ids']}"


@pytest.mark.tokenizer
def test_list_tokenization(tokenizer: SpokenDialogTokenizer):
    text_string = "Hello there how are you today?"
    text_list = [text_string, text_string, text_string]
    ans = "<ts> ".join(text_list)

    t = tokenizer(text_list, include_end_ts=False, include_pre_space=False)
    ts_pred = tokenizer.decode(t["input_ids"])
    assert ts_pred == ans, f"Expected {ans} but got {ts_pred}"

    t = tokenizer(text_list, include_end_ts=True, include_pre_space=False)
    ts_pred = tokenizer.decode(t["input_ids"])
    assert ts_pred == ans + "<ts>", f"Expected {ans} but got {ts_pred}"

    t = tokenizer(text_list, include_end_ts=False, include_pre_space=True)
    ts_pred = tokenizer.decode(t["input_ids"])
    assert ts_pred == " " + ans, f"Expected {ans} but got {ts_pred}"

    t = tokenizer(text_list, include_end_ts=True, include_pre_space=True)
    ts_pred = tokenizer.decode(t["input_ids"])
    assert ts_pred == " " + ans + "<ts>", f"Expected {ans} but got {ts_pred}"


@pytest.mark.tokenizer
def test_list_tokenization(tokenizer: SpokenDialogTokenizer):
    text_string = "Hello there how are you today?"
    list_of_list_of_strings = [
        [text_string, text_string],
        [text_string, text_string, text_string],
    ]
    ans_string = "hello there how are you today"
    ans0 = "<ts> ".join([ans_string, ans_string]).lower()
    ans1 = "<ts> ".join([ans_string, ans_string, ans_string]).lower()

    t = tokenizer(
        list_of_list_of_strings, include_end_ts=False, include_pre_space=False
    )
    ts_pred0 = tokenizer.decode(t["input_ids"][0])
    ts_pred1 = tokenizer.decode(t["input_ids"][1])
    assert ts_pred0 == ans0, f"Expected {ans0} but got {ts_pred0}"
    assert ts_pred1 == ans1, f"Expected {ans1} but got {ts_pred1}"

    t = tokenizer(list_of_list_of_strings, include_end_ts=True, include_pre_space=False)
    ts_pred0 = tokenizer.decode(t["input_ids"][0])
    ts_pred1 = tokenizer.decode(t["input_ids"][1])
    assert ts_pred0 == ans0 + "<ts>", f"Expected {ans0} but got {ts_pred0}"
    assert ts_pred1 == ans1 + "<ts>", f"Expected {ans1} but got {ts_pred1}"

    t = tokenizer(list_of_list_of_strings, include_end_ts=False, include_pre_space=True)
    ts_pred0 = tokenizer.decode(t["input_ids"][0])
    ts_pred1 = tokenizer.decode(t["input_ids"][1])
    assert ts_pred0 == " " + ans0, f"Expected {ans0} but got {ts_pred0}"
    assert ts_pred1 == " " + ans1, f"Expected {ans1} but got {ts_pred1}"

    t = tokenizer(list_of_list_of_strings, include_end_ts=True, include_pre_space=True)
    ts_pred0 = tokenizer.decode(t["input_ids"][0])
    ts_pred1 = tokenizer.decode(t["input_ids"][1])
    assert ts_pred0 == " " + ans0 + "<ts>", f"Expected {ans0} but got {ts_pred0}"
    assert ts_pred1 == " " + ans1 + "<ts>", f"Expected {ans1} but got {ts_pred1}"

    # Test with return_tensors
    t = tokenizer(
        list_of_list_of_strings,
        include_end_ts=False,
        include_pre_space=False,
        return_tensors="pt",
    )

    ts_pred0 = tokenizer.decode(t["input_ids"][0][0])
    ts_pred1 = tokenizer.decode(t["input_ids"][1][0])
    assert ts_pred0 == ans0, f"Expected {ans0} but got {ts_pred0}"
    assert ts_pred1 == ans1, f"Expected {ans1} but got {ts_pred1}"


@pytest.mark.tokenizer
def test_word_probs_tensor(tokenizer: SpokenDialogTokenizer):
    text_list = [
        "Yesterday i had tommorows intervention",
        "Oh is that so but yesterday?",
    ]
    ans_probs = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Tensors
    t = tokenizer(text_list, return_tensors="pt")

    # Get the input_ids (disregard batch dimension)
    input_ids = t["input_ids"][0]

    # Fix the probabilities
    probs = torch.arange(len(input_ids))

    # Extract the word probabilities
    p = tokenizer.extract_word_probs(input_ids, probs)
    assert len(p["words"]) == len(
        p["probs"]
    ), f"Expected the same length but got words: {len(p['words'])} and probs: {len(p['probs'])}"
    assert isinstance(
        p["probs"], torch.Tensor
    ), f"Expected a probs to be a tensor. got {type(p['probs'])}"
    assert (
        p["probs"].tolist() == ans_probs
    ), f"Expected {ans_probs} but got {p['probs']}"


@pytest.mark.tokenizer
def test_word_probs_list(tokenizer: SpokenDialogTokenizer):
    text_list = [
        "Yesterday i had tommorows intervention",
        "Oh is that so but yesterday?",
    ]
    ans_probs = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Tensors
    t = tokenizer(text_list)

    # Get the input_ids
    input_ids = t["input_ids"]

    # Fix the probabilities
    probs = torch.arange(len(input_ids)).tolist()

    # Extract the word probabilities
    p = tokenizer.extract_word_probs(input_ids, probs)
    assert len(p["words"]) == len(
        p["probs"]
    ), f"Expected the same length but got words: {len(p['words'])} and probs: {len(p['probs'])}"
    assert isinstance(
        p["probs"], torch.Tensor
    ), f"Expected a probs to be a tensor. got {type(p['probs'])}"
    assert p["probs"] == ans_probs, f"Expected {ans_probs} but got {p['probs']}"


# @pytest.mark.tokenizer
# def test_idx_to_tokens(tokenizer: SpokenDialogTokenizer):
#     tok_out = tokenizer(turn_list, include_end_ts=False)
#     ids_list = tok_out["input_ids"]
#     ids_tens = torch.tensor(tok_out["input_ids"])
#     t1 = tokenizer.idx_to_tokens(ids_list)
#     t2 = tokenizer.idx_to_tokens(ids_tens)
#     t3 = tokenizer.idx_to_tokens(ids_list[0])
#     assert t1 == t2
#     assert isinstance(t3, str)


# @pytest.mark.tokenizer
# def test_idx_tensor_to_tokens(tokenizer: SpokenDialogTokenizer):
#     tok_out = tokenizer(turn_list, include_end_ts=False)
#     ids_tens = torch.tensor(tok_out["input_ids"])
#     t2 = tokenizer.idx_to_tokens(ids_tens)
