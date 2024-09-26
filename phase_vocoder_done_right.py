import numpy as np
from scipy.fftpack import fft, rfft,ifft, irfft,fftfreq
import librosa
import soundfile as sf
from tqdm import tqdm

def delta_for(m, phi, phi1, R_a, N=8192):
    delta = phi1 - phi - 2*np.pi*m*R_a/N
    delta = delta % np.pi
    return 1 / R_a * delta + 2*np.pi*m/N

def delta_bck(m, phi, phi1, R_a, N=8192):
    delta = phi - phi1 - 2 * np.pi * m * R_a / N
    delta = delta % np.pi
    return 1 / R_a * delta + 2 * np.pi * m / N


def delta_cent(m, phi, phi1, phi_1, R_a, N=8192):
    return 1/2 * (delta_bck(m, phi, phi_1, R_a, N) + delta_for(m,phi,phi1, R_a, N))

def PV(x, N=8192, R_s=1024, stretch=1, tol=10**(-6)):

    R_a = int(R_s/stretch)

    nb_fen = int((len(x)-int(N - R_s)) / R_a)
    x_fin = np.zeros(int(len(x)//stretch))
    print("temps d'attente estimé : ", (nb_fen*5)//60, "minutes, ou ", (nb_fen*5)/3600, "heures")
    for i in tqdm(range(nb_fen - 2)):  # attention normalement -1 (on fait pas les derniers)
        xn1 = np.copy(x[i * R_a: i * R_a + N])
        xn = np.copy(x[i * R_a + N: i * R_a + 2 * N])
        if len(xn) != N:
            xn = np.pad(xn, (0, N - len(xn)), mode='constant')
        if len(xn1) != N:
            xn1 = np.pad(xn1, (0, N - len(xn1)), mode='constant')
        xn1 *= np.hanning(N)
        xn *= np.hanning(N)

        # passage dans fourier
        xn1_fou = fft(xn1)
        s1 = np.absolute(xn1_fou)
        phi1 = np.angle(xn1_fou)
        xn_fou = fft(xn)
        s = np.absolute(xn_fou)
        phi = np.angle(xn_fou)

        abstol = tol * max(s.all(), s1.all())
        I_1 = np.zeros((len(s),3))
        M = 0
        for m in range(len(s)):
            if s[m] > abstol:
                # 1re colonne donne le max, 2e donne le m donc I, 3e donne n
                I_1[m,0], I_1[m,1], I_1[m,2] = s[m], m, i
                M+=1
            else:
                phi[m] = np.random.random()*np.pi
        # heap
        heap = np.array(I_1)
        I = list(I_1[:, 1])
        for m in I:
            heap = np.insert(heap, -1, [s1[int(m)], m, i-1], axis=0)

        # lancement de l'algorithme
        while len(I) > 0:
            k = np.argmax(heap[:,0])
            m, n = heap[k,1], heap[k,2]
            heap = np.delete(heap, k, axis=0)
            if n == i-1:
                if m in I:
                    m = int(m)
                    phi[m] = phi1[m] + R_s/2 * (delta_for(m,phi1[m],phi[m],R_a,N=N) + delta_bck(m,phi[m],phi1[m],R_a,N=N))
                    I.remove(m)
                    heap = np.insert(heap, -1, [s[m], m, i], axis=0)
            elif n == i:
                if m+1 in I:
                    m = int(m)
                    phi[m+1] = phi[m] + R_s/2 * (delta_cent(m,phi[m],phi[(m+1)%N],phi[(m-1)%N], R_a=R_a, N=N) + delta_cent(m,phi[m+1],phi[(m+2)%N],phi[m],R_a=R_a, N=N))
                    I.remove(m+1)
                    heap = np.insert(heap, -1, [s[m+1], m+1, i], axis=0)
                if m-1 in I:
                    m = int(m)
                    phi[m-1] = phi[m] + R_s / 2 * (delta_cent(m, phi[m], phi[(m+1)%N], phi[m-1], R_a=R_a, N=N) + delta_cent(m, phi[m-1], phi[m],phi[m-2], R_a=R_a, N=N))
                    I.remove(m-1)
                    heap = np.insert(heap, -1, [s[m-1], m-1, i], axis=0)
        xn = ifft(s * np.exp(1j * phi))
        xn = np.real(xn)
        x_fin[R_s * i: R_s * i + N] += xn
    return x_fin

audio, fs = librosa.load("C:/Users/user/PycharmProjects/Stage_Vocoder_FRAGA/biblio_son/the_intruder_short.wav",sr=None)
print(len(audio))
audio_modify = PV(audio)
sf.write('pv_test_intruder.wav', audio_modify, fs)