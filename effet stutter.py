import soundfile as sf
import pytsmod

audio, fs = sf.read("C:/Users/user/PycharmProjects/Stage_Vocoder_FRAGA/biblio_son/Paint_It_Black_mono.wav")

def artefact(x, alpha, type='PV'):
    if type == 'PV':
        audio_modif = pytsmod.phase_vocoder(audio, alpha)
        audio_modif = pytsmod.phase_vocoder(audio_modif, 1/alpha)
    elif type == 'WSOLA':
        audio_modif = pytsmod.wsola(audio, alpha)
        audio_modif = pytsmod.wsola(audio_modif, 1 / alpha)
    elif type == 'OLA':
        audio_modif = pytsmod.ola(audio, alpha)
        audio_modif = pytsmod.ola(audio_modif, 1 / alpha)
    else:
        print("Error : Unknown Type, Please choose one of these next propositions : 'PV', 'WSOLA', 'OLA'",)
    return audio_modif
sf.write('paint_it_pv_10.wav', artefact(audio,10), fs)


