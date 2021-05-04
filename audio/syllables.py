import pronouncing
from pronouncing import phones_for_word, syllable_count as pron_syll_count

from nltk import download as nltk_download
from nltk.corpus import cmudict


def backup_syllables(word):
    """
    Referred from:
        stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    from:
        https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
    """
    count = 0
    vowels = "aeiouy"
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count


class SyllableCountPronounciation(object):
    def __call__(self, word):
        try:
            phones = pronouncing.phones_for_word(word)[0]
            return pron_syll_count(phones)
        except IndexError:
            return backup_syllables(word)


class SyllableCountNLTK(object):
    def __init__(self):
        try:
            self.cmu_dict = cmudict.dict()
        except LookupError:
            nltk_download("cmudict")
            self.cmu_dict = cmudict.dict()

    def __call__(self, word):
        try:
            return [
                len(list(y for y in x if y[-1].isdigit()))
                for x in self.cmu_dict[word.lower()]
            ]
        except KeyError:
            return backup_syllables(word)


if __name__ == "__main__":
    import time

    s = "explosion"
    s = "pronounciation"

    syl_ntlk = SyllableCountNLTK()
    syl_pron = SyllableCountNLTK()

    n = syl_ntlk(s)
    print(f"NLTK: {s} -> {n}")

    n = syl_pron(s)
    print(f"Pron: {s} -> {n}")

    text = "I made Pronouncing because I wanted to be able to use the CMU Pronouncing Dictionary in my projects and teach other people how to use it without having to install the grand behemoth that is NLTK"

    t = time.time()
    for i in range(100000):
        for s in text.split(" "):
            n = syl_ntlk(s)  # almost as fast as pron
    t = time.time() - t
    print("NLTK: ", round(t, 3))

    t = time.time()
    for i in range(100000):
        for s in text.split(" "):
            n = syl_pron(s)  # fastest
    t = time.time() - t
    print("Pron: ", round(t, 3))
