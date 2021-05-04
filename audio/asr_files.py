# Imports the Google Cloud client library
from google.cloud import speech
import io
import os
from ttd.utils import write_json

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/erik/projects/data/GOOGLE_SPEECH_CREDENTIALS.json"


class ASR(object):
    def __init__(self, sr=16000):
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sr,
            language_code="en-US",
            enable_word_time_offsets=True,
        )
        self.client = speech.SpeechClient()

    def format_response(self, response):
        turn = {
            "text": response.results[0].alternatives[0].transcript,
            "starts": [],
            "ends": [],
            "speaker_id": 0,
        }
        for word_data in response.results[0].alternatives[0].words:
            w = word_data.word
            start_time = word_data.start_time.total_seconds()
            end_time = word_data.end_time.total_seconds()
            turn["starts"].append(start_time)
            turn["ends"].append(end_time)
        turn["start"] = turn["starts"][0]
        return turn

    def transcribe_file(self, speech_file):
        """Transcribe the given audio file."""

        with io.open(speech_file, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        response = self.client.recognize(config=self.config, audio=audio)

        return self.format_response(response)


if __name__ == "__main__":

    from tqdm import tqdm
    from glob import glob

    asr = ASR()

    root = "data/sample_phrases"
    files = glob(os.path.join(root, "*.wav"))

    for speech_file in tqdm(files, desc="transcribe files"):
        root = os.path.split(speech_file)[0]
        name = os.path.basename(speech_file)
        turn = asr.transcribe_file(speech_file)
        write_json([turn], os.path.join(root, name.replace(".wav", ".json")))
