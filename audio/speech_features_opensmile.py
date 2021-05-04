import opensmile
import time

import torchaudio
import torchaudio.transforms as AT

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sounddevice as sd

    smile = opensmile.Smile(
        # feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        # feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=2,
        num_workers=4,
        verbose=True,
    )
    smile.feature_names

    sr = 16000
    wav_path = "/home/erik/projects/data/maptask/audio/q6ec2.wav"
    waveform, tmp_sr = torchaudio.load(wav_path)
    if tmp_sr != sr:
        resampler = AT.Resample(orig_freq=tmp_sr, new_freq=sr)
        waveform = resampler(waveform)

    fig, ax = plt.subplots(2, 1)

    T = 10
    # wav = waveform[:, int(T*sr):int(2*T*sr)]
    s = 158000
    wav = waveform[:, s : s + 16000]
    t = time.time()
    # y = smile.process_file(wav_path)
    y = smile.process_signal(wav, sr)  # returns pandas.core.frame.DataFrame
    print(round(time.time() - t, 2))
    sd.play(wav[0], samplerate=sr)
    for a in ax:
        a.cla()
    ch = 1
    f0 = y[f"F0final_sma-{ch}"].to_numpy()
    v = y[f"voicingFinalUnclipped_sma-{ch}"].to_numpy()
    ax[0].plot(f0)
    ax[0].plot(v)
    # ax[0].plot(y[f'hammarbergIndex_sma3-{ch}'].to_numpy())
    # ax[0].plot(y[f'Loudness_sma3-{ch}'].to_numpy())
    # ax[0].plot(y[f'slope0-500_sma3-{ch}'].to_numpy())
    # ax[0].plot(y[f'slope500-1500_sma3-{ch}'].to_numpy())
    ax[1].plot(wav[ch])
    plt.tight_layout()
    plt.pause(0.01)
