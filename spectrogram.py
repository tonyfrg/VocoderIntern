import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy import signal
from scipy.fftpack import fft, rfft,ifft, irfft,fftfreq
import matplotlib.pyplot as plt
from IPython.display import Audio, Video, YouTubeVideo
from skimage import util
import librosa

import os

'''
Le spectrogramme ci-dessus fonctionne, mais il ne montre pas comment il est généré.
Je vais tenter cet après-midi de m'en charger ! (en regardant le code de mes homologues).
Après réflexion, je me dit qu'il faudrait que je m'en fabrique un avec mes propres 
connaissances plus tard ...
'''


def plot_spectro(audio, fs, titre='', taille_fenetre=1024, recouvrement=None, normalize=True):
    N = audio.shape[0]
    L = N / fs  # ordre de grandeur pour le temps

    if recouvrement is None:
        recouvrement = taille_fenetre // 2

    slices = util.view_as_windows(audio, window_shape=taille_fenetre, step=recouvrement)

    win = np.hanning(taille_fenetre + 1)[:-1]
    slices = slices * win

    slices = slices.T

    spectrum = np.fft.fft(slices, axis=0)[:taille_fenetre // 2 + 1:-1]

    f, ax = plt.subplots(figsize=(20, 3))

    S = np.abs(spectrum)
    S = 20 * np.log10(S / np.max(S))  # conversion en dB

    if normalize:
        S = (S - np.min(S)) / (np.max(S) - np.min(S))

    pos = ax.imshow(S, origin='lower', cmap='viridis', extent=(0, L, 0, fs / 2 / 1000))
    ax.set_title(titre)
    ax.axis('tight')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [s]')
    cbar = f.colorbar(pos, ax=ax)
    plt.plot()
son = 'C:/Users/user/PycharmProjects/stage_py/Anthony/biblio_son/grieg_good_morning.wav'
fs, x = wavfile.read(son)
spectrogram(x, fs, 'nightingale')
son_2 = 'C:/Users/user/PycharmProjects/stage_py/Anthony/biblio_son/grieg_good_morning.wav'
fs_2, x_2 = wavfile.read(son_2)
x_2 = x_2.astype(np.float32)[:,0]
plot_spectro(x_2, fs_2, 'nightingale')
plt.show()
Audio(x_2, rate=fs_2)
