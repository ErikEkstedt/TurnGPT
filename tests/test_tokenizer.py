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
@pytest.mark.parametrize(
    "text_list, ans_words, ans_probs",
    [
        (
            [
                "Yesterday i had tommorows intervention",
                "Oh is that so but yesterday?",
                "I don't know",
            ],
            [
                "yesterday",
                "i",
                "had",
                "tommorows",
                "intervention",
                "<ts>",
                "oh",
                "is",
                "that",
                "so",
                "but",
                "yesterday",
                "<ts>",
                "i",
                "don't",
                "know",
            ],
            [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20],
        ),
        (
            [
                "Yesterday i had tommorows intervention",
                "Oh is that so but yesterday?",
            ],
            [
                "yesterday",
                "i",
                "had",
                "tommorows",
                "intervention",
                "<ts>",
                "oh",
                "is",
                "that",
                "so",
                "but",
                "yesterday",
            ],
            [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        (
            [
                "Yesterday i had tommorows intervention",
            ],
            ["yesterday", "i", "had", "tommorows", "intervention"],
            [1, 2, 3, 7, 8],
        ),
    ],
)
def test_word_probs_list(
    text_list, ans_words, ans_probs, tokenizer: SpokenDialogTokenizer
):
    t = tokenizer(text_list, include_end_ts=False)

    # Get the input_ids
    input_ids = t["input_ids"]

    # Fix the probabilities
    probs = torch.arange(len(input_ids))

    # Extract the word probabilities
    wp = tokenizer.extract_word_probs(input_ids, probs)
    w, p = wp["words"], wp["probs"]
    assert len(w) == len(
        p
    ), f"Expected the same length but got words: {len(w)} and probs: {len(p)}"
    assert isinstance(
        p, torch.Tensor
    ), f"Expected a probs to be a tensor. got {type(p)}"
    assert p.tolist() == ans_probs, f"Expected {ans_probs} but got {p}"
    assert w == ans_words, f"Expected {ans_words} but got {w}"


@pytest.mark.tokenizer
@pytest.mark.parametrize(
    "text_list, ans_words, ans_probs",
    [
        (
            [
                "Yesterday i had tommorows intervention",
                "Oh is that so but yesterday?",
                "I don't know",
            ],
            ["i", "don't", "know"],
            [17, 19, 20],
        ),
        (
            [
                "Yesterday i had tommorows intervention",
                "Oh is that so but yesterday?",
            ],
            ["oh", "is", "that", "so", "but", "yesterday"],
            [10, 11, 12, 13, 14, 15],
        ),
        (
            [
                "Yesterday i had tommorows intervention",
            ],
            ["yesterday", "i", "had", "tommorows", "intervention"],
            [1, 2, 3, 7, 8],
        ),
    ],
)
def test_word_probs_filter_last_utterance(
    text_list, ans_words, ans_probs, tokenizer: SpokenDialogTokenizer
):
    # Use case: use the model in a SDS to infer upcoming TRPs
    # Assume that we don't add <ts> tokens to the end of the utterances
    t = tokenizer(text_list, include_end_ts=False, return_tensors="pt")  # (1, N)
    p = torch.arange(t["input_ids"].shape[-1]).unsqueeze(0)
    wp = tokenizer.extract_word_probs(t["input_ids"].squeeze(0), p.squeeze(0))
    ww, pp = tokenizer.word_probs_filter_last_utterance(wp["words"], wp["probs"])
    assert ww == ans_words, f"Expected {ans_words} but got {ww}"
    assert pp.tolist() == ans_probs, f"Expected {ans_probs} but got {pp}"


@pytest.mark.tokenizer
@pytest.mark.parametrize(
    "projections, ans_n_words, ans_prefix",
    [
        (
            [
                " what's going on<ts> everything",
                " what's wrong<ts> there's",
                " what's going on<ts> oh",
                " what you're doing here<ts>",
                "<ts> i don't know what",
                " what's going on<ts> nothing",
                " what's going on<ts> i",
                " what's wrong<ts> i don",
                " what's wrong<ts> i don",
                " what's going on<ts> i",
                " what's going on<ts> i",
                " what's going on you look",
                " what's going on<ts> my",
                " what you're doing here<ts>",
                " about yourself<ts> well i'm",
                "<ts> i was just wondering why",
                " what's going on<ts> there",
                "<ts> oh just a bit of",
                " what's going on<ts> i",
                " what's wrong<ts> oh nothing",
            ],
            [3, 2, 3, 4, 0, 3, 3, 2, 2, 3, 3, 5, 3, 4, 2, 0, 3, 0, 3, 2],
            [
                "everything",
                "there's",
                "oh",
                "i don't know what",
                "nothing",
                "i",
                "i don",
                "i don",
                "i",
                "i",
                "my",
                "well i'm",
                "i was just wondering why",
                "there",
                "oh just a bit of",
                "i",
                "oh nothing",
            ],
        )
    ],
)
def test_prefix_and_word_counter(
    projections, ans_n_words, ans_prefix, tokenizer: SpokenDialogTokenizer
):
    nwords, prefix = tokenizer.get_prefixes(projections)
    assert nwords == ans_n_words, f"Expected {ans_n_words} but got {nwords}"
    assert prefix == ans_prefix, f"Expected {ans_prefix} but got {prefix}"

    nw = torch.tensor(nwords)
    assert (
        nw <= 2
    ).sum() == 8, f"Epected 8 prefixes with less than 2 words but got {nw}"
    assert (
        nw <= 3
    ).sum() == 17, f"Epected 17 prefixes with less than 3 words but got {nw}"
    assert (
        nw <= 4
    ).sum() == 19, f"Epected 19 prefixes with less than 4 words but got {nw}"
    assert (
        nw <= 5
    ).sum() == 20, f"Epected 20 prefixes with less than 5 words but got {nw}"
