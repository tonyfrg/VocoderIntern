import pygame
from pygame import mixer
import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import pytsmod
import soundfile as sf

##### Fonction nécessaire pour le piano #####

def pitch(x, a):
    y = pytsmod.phase_vocoder(x, a)
    yy = signal.resample(y, x.size)
    out = np.array((yy,yy)).T
    return out

# 2 variables utiles #
alpha = 1  # permet de passer d'une note à l'autre
p = 2**(1/12)  # rapport entre 2 demi-tons

### importing sound
file = r'C:/Users/user/PycharmProjects/Stage_Vocoder_FRAGA/biblio_son/discours_abbe_pierre_mono.wav'
audio, fs = sf.read(file, dtype='int16')

# passage en stéréo (très chiant pygame ouuuu)
l = len(audio)
audio_stereo = np.zeros((l,2), dtype=np.int16)
audio_stereo[:,0], audio_stereo[:,1] = audio, audio

### Calculs des différents pitch possibles ###
audio_C = pitch(audio, 1)
audio_Db = pitch(audio, p)
audio_D = pitch(audio, p**2)
audio_Eb = pitch(audio, p**3)
audio_E = pitch(audio, p**4)
audio_F = pitch(audio, p**5)
audio_Gb = pitch(audio, p**6)
audio_G = pitch(audio, p**7)
audio_Ab = pitch(audio, p**8)
audio_A = pitch(audio, p**9)
audio_Bb = pitch(audio, p**10)
audio_B = pitch(audio, p**11)
audio_CC = pitch(audio, p**12)

### Démarrage du Programme ###
pygame.init()
window = pygame.display.set_mode((300, 300))
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
            t = pygame.time.get_ticks()
            mixer.stop()
            if event.key == pygame.K_a:
                # C
                audio_stereo[int(fs * t / 1000):] = audio_C[int(fs * t / 1000):]
            elif event.key == pygame.K_2:
                # C#
                audio_stereo[int(fs * t / 1000):] = audio_Db[int(fs * t / 1000):]
            elif event.key == pygame.K_z:
                # D
                audio_stereo[int(fs * t / 1000):] = audio_D[int(fs * t / 1000):]
            elif event.key == pygame.K_3:
                # D#
                audio_stereo[int(fs * t / 1000):] = audio_Eb[int(fs * t / 1000):]
            elif event.key == pygame.K_e:
                # E
                audio_stereo[int(fs * t / 1000):] = audio_E[int(fs * t / 1000):]
            elif event.key == pygame.K_r:
                # F
                audio_stereo[int(fs * t / 1000):] = audio_F[int(fs * t / 1000):]
            elif event.key == pygame.K_5:
                # F#
                audio_stereo[int(fs * t / 1000):] = audio_Gb[int(fs * t / 1000):]
            elif event.key == pygame.K_t:
                # G
                audio_stereo[int(fs * t / 1000):] = audio_G[int(fs * t / 1000):]
            elif event.key == pygame.K_6:
                # G#
                audio_stereo[int(fs * t / 1000):] = audio_Ab[int(fs * t / 1000):]
            elif event.key == pygame.K_y:
                # A
                audio_stereo[int(fs * t / 1000):] = audio_A[int(fs * t / 1000):]
            elif event.key == pygame.K_2:
                # A#
                audio_stereo[int(fs * t / 1000):] = audio_Bb[int(fs * t / 1000):]
            elif event.key == pygame.K_u:
                # B
                audio_stereo[int(fs * t / 1000):] = audio_B[int(fs * t / 1000):]
            elif event.key == pygame.K_i:
                # C octave plus
                audio_stereo[int(fs * t / 1000):] = audio_CC[int(fs * t / 1000):]
            elif event.key == pygame.K_x:
                # Stop the music playback
                mixer.music.stop()
                print("music is stopped....")
                pygame.quit()
                break
            sound = mixer.Sound(audio_stereo[int(fs * t / 1000):])
            mixer.Sound.play(sound)
    window.fill(0)

sf.write('talkbox.wav', audio_stereo, 44100, 'PCM24')