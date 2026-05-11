import librosa
import numpy as np

def extract_mel_spectrogram(audio_path):
    """
    Extracts Mel Spectrogram from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def extract_mfcc(audio_path, n_mfcc=13):
    """
    Extracts MFCC features.
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

if __name__ == "__main__":
    # Example usage
    # features = extract_mfcc("data/samples/sample_audio.wav")
    # print(features.shape)
    pass
