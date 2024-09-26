import pygame
from pygame import mixer
import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import pytsmod

##### Fonctions nécessaires pour le piano #####
def WSOLA(x, N=1024, stretch=2):
    x_stretch = np.zeros(int(len(x) // stretch))

    Hs = int(N / 2)
    Ha = stretch * Hs

    if stretch < 1:
        delta_max = int(Ha)
    else:
        delta_max = int(N // 2)

    deltam = 0
    i = 0

    while (i + 1) * int(Ha) + N + delta_max < len(x):
        xm0 = np.copy(x[i * int(Ha) + deltam: i * int(Ha) + deltam + N])
        #print(xm0.size)
        #print(xm0)
        xm0 = xm0 * np.hanning(len(xm0))
        #xm0 = xm0 * np.hanning(int(xm0.size))
        x_stretch[Hs * i : Hs * i + N] += xm0

        # recollage
        xm0_tilde = np.copy(x[i * int(Ha) + deltam + Hs : i * int(Ha) + deltam + Hs + N])
        xm1_plus = np.copy(x[(i + 1) * int(Ha) - delta_max : (i + 1) * int(Ha) + N + delta_max])
        corr = np.correlate(xm1_plus, xm0_tilde)
        deltam = np.argmax(corr) - len(corr) // 2

        i += 1
    return x_stretch

def pitch(x, a):
    y = pytsmod.phase_vocoder(x, a)
    yy = signal.resample(y, x.size)
    return yy

# 2 variables utiles #
alpha = 1  # permet de passer d'une note à l'autre
p = 2**(1/12)  # rapport entre 2 demi-tons

### importing sound
file = r'C:/Users/user/PycharmProjects/Stage_Vocoder_FRAGA/biblio_son/'
fs, audio = wave.read(file)

# passage en stéréo (très chiant pygame ouuuu)
l = len(audio)
audio_stereo = np.zeros((l,2), dtype=np.int16)
audio_stereo[:,0] = audio

### Démarrage du Programme ###
pygame.init()
window = pygame.display.set_mode((300, 300))

bits = 16
#the number of channels specified here is NOT
#the channels talked about here http://www.pygame.org/docs/ref/mixer.html#pygame.mixer.get_num_channels
mixer.pre_init(44100, -bits, 1, allowedchanges=0)
mixer.init()
sound = pygame.sndarray.make_sound(audio_stereo)
print(f'{mixer.get_init()=}')
# lance la musique
mixer.Sound.play(sound)

# Infinite loop
while True:
    # take user input
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            t = pygame.time.get_ticks() % (l * 1000 / fs)
            mixer.stop()
            if event.key == pygame.K_a:
                # C
                audio_stereo[int(fs * t / 1000):,0] = pitch(audio_stereo[int(fs * t / 1000):,0], p ** 0 / alpha)
                alpha = p ** 0
            elif event.key == pygame.K_2:
                # C#
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 1 / alpha)
                alpha = p ** 1
            elif event.key == pygame.K_z:
                # D
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 2 / alpha)
                alpha = p ** 2
            elif event.key == pygame.K_3:
                # D#
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 3 / alpha)
                alpha = p ** 3
            elif event.key == pygame.K_e:
                # E
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 4 / alpha)
                alpha = p ** 4
            elif event.key == pygame.K_r:
                # F
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 5 / alpha)
                alpha = p ** 5
            elif event.key == pygame.K_5:
                # F#
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 6 / alpha)
                alpha = p ** 6
            elif event.key == pygame.K_t:
                # G
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 7 / alpha)
                alpha = p ** 7
            elif event.key == pygame.K_6:
                # G#
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 8 / alpha)
                alpha = p ** 8
            elif event.key == pygame.K_y:
                # A
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 9 / alpha)
                alpha = p ** 9
            elif event.key == pygame.K_2:
                # A#
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 10 / alpha)
                alpha = p ** 10
            elif event.key == pygame.K_u:
                # B
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 11 / alpha)
                alpha = p ** 11
            elif event.key == pygame.K_i:
                # C octave plus
                audio_stereo[int(fs * t / 1000):, 0] = pitch(audio_stereo[int(fs * t / 1000):, 0], p ** 12 / alpha)
                alpha = p ** 12
            elif event.key == pygame.K_x:
                # Stop the music playback
                mixer.music.stop()
                print("music is stopped....")
                pygame.quit()
                exit()
            sound = mixer.Sound(audio_stereo[int(fs * t / 1000):])
            mixer.Sound.play(sound)
    window.fill(0)

sf.write('talkbox',sound)