# audio-to-spectrogram
Convert audio files into spectrograms using Python and the Librosa library.
# Audio to Spectrogram

This project demonstrates how to convert audio files into spectrograms using the Python library [Librosa](https://librosa.org/).

## Features

- Load audio files (.wav, .mp3)
- Generate mel spectrograms
- Visualize audio features
- Prepare audio data for machine learning

## Example Code

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("audio_sample.wav")
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
