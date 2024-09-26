import numpy as np
from scipy.fftpack import fft, rfft,ifft, irfft,fftfreq
import librosa
import soundfile as sf
from tqdm import tqdm


def PV(x, N=8192, R_s=1024, stretch=1, tol=10**(-6)):
    R_a = int(R_s/stretch)
    nb_fen = int((len(x)-int(N - R_s)) / R_a)
    PHI = []  # liste des phases de chaque rang n_i
    S = []
    x_fin = np.zeros(int(len(x)//stretch))
    for i in range(nb_fen - 1):  # attention normalement -1 (on fait pas les derniers)
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
    PHI_S = [np.zeros((N,1)) for i in range(len(PHI))]
    PHI_S[0] = PHI[0]
    xn = ifft(S[0] * np.exp(1j * PHI_S[0]))
    xn = np.real(xn)
    x_fin[0:N] += xn[0]
    for n in tqdm(range(1, nb_fen - 1)):
        for k in range(len(PHI_S)):
            PHI_S[n][k] = PHI_S[n-1][k] + stretch * np.fmod(PHI[n][k] - PHI[n-1][k] + (2*k*np.pi)/N, 2*np.pi) - 2*k*np.pi/N
        xn = ifft(S[n] * np.exp(1j * PHI_S[n]))
        xn = np.real(xn)
        x_fin[R_s * n: R_s * n + N] += xn[n]
    return x_fin

audio, fs = librosa.load("C:/Users/user/PycharmProjects/stage_py/Anthony/biblio_son/grieg_very_short.wav",sr=None)
print(len(audio))
audio_modify = PV(audio,)
print(max(audio_modify))
print(audio_modify)
sf.write('pv2_test_grieg.wav', audio_modify, fs)