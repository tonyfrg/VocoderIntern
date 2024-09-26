import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, rfft,ifft, irfft,fftfreq
import librosa
import soundfile as sf
from tqdm import tqdm

# 2**11=2048, 2**14=8192, 2**8=256
def PV(x, N=2**14, R_s=2**10, stretch=1, graph=False):
    R_a = int(R_s/stretch)
    nb_fen = int((len(x)-int(N - R_s)) / R_a)
    PHI = []  # liste des phases de chaque rang n_i
    S = []
    x_fin = np.zeros(int(len(x)//stretch))
    for i in tqdm(range(nb_fen - 1)):  # attention normalement -1 (on fait pas les derniers)
        xn = np.copy(x[i * R_a: i * R_a + N])
        if len(xn) != N:
            xn = np.pad(xn, (0, N - len(xn)), mode='constant')
        xn *= np.hanning(N)
        # passage dans fourier
        xn_fou = fft(xn)
        s = np.absolute(xn_fou)
        phi = np.angle(xn_fou)
        S.append(s)
        PHI.append(phi)
    PHI_S = [np.zeros((N, 1)) for i in range(len(PHI))]
    PHI_S[0] = np.fmod(PHI[0], np.pi)
    xn = ifft(S[0] * np.exp(1j * PHI_S[0]))
    xn = np.real(xn)
    xn *= np.hanning(N)
    x_fin[0:N] += xn[0]
    for n in tqdm(range(1, nb_fen - 2)):
        PHI_S[n] = np.fmod(PHI_S[n-1] + stretch/2 * (np.fmod(PHI[n]-PHI[n-1], np.pi) + np.fmod(PHI[n+1]-PHI[n], np.pi)), np.pi)
        xn = ifft(S[n].T * np.exp(1j * PHI_S[n]))
        xn = np.real(xn)
        xn *= np.hanning(N)
        x_fin[R_s * n: R_s * n + N] += xn
    PHI_S[-1] = PHI_S[-2] + stretch * np.fmod(PHI[-1] - PHI[-2], np.pi)
    xn = ifft(S[-1].T * np.exp(1j * PHI_S[-1]))
    xn = np.real(xn)
    xn *= np.hanning(N)
    x_fin[-N - 1: -1] += xn
    return x_fin

audio, fs = librosa.load("C:/Users/user/PycharmProjects/Stage_Vocoder_FRAGA/biblio_son/the_intruder_short.wav",sr=None)
print(len(audio))
audio_modify = PV(audio, graph=False)
print(max(audio_modify))
print(audio_modify)
sf.write('pv3_test_intruder.wav', audio_modify, int(fs))
