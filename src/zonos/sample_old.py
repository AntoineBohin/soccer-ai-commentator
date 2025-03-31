import os
import shutil

# Définir le PATH pour inclure le répertoire contenant espeak
os.environ["PATH"] = os.path.join(os.environ["HOME"], "src/local/bin") + ":" + os.environ.get("PATH", "")

# Définir explicitement le chemin vers l'exécutable espeak
os.environ["ESPEAK_PATH"] = os.path.join(os.environ["HOME"], "src/local/bin", "espeak")

# Définir PHONEMIZER_ESPEAK_PATH (le dossier où se trouve espeak)
os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(os.environ["HOME"], "local/bin", "espeak")

# Définir PHONEMIZER_ESPEAK_LIBRARY
# Pour Linux, cela correspond généralement à une bibliothèque partagée (.so).
# Vous devez adapter le chemin selon l'endroit où la bibliothèque est installée.
# Par exemple, si vous avez compilé eSpeak NG avec --prefix=$HOME/local,
# la bibliothèque pourrait se trouver dans $HOME/local/lib/libespeak-ng.so.
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(os.environ["HOME"], "local/lib", "libespeak-ng.so")

# Vérification des chemins
print("PATH:", os.environ["PATH"])
print("ESPEAK_PATH:", os.environ["ESPEAK_PATH"])
print("PHONEMIZER_ESPEAK_PATH:", os.environ["PHONEMIZER_ESPEAK_PATH"])
print("PHONEMIZER_ESPEAK_LIBRARY:", os.environ["PHONEMIZER_ESPEAK_LIBRARY"])
print("Chemin de espeak via shutil:", shutil.which("espeak"))

# Ensuite, vos imports habituels
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

wav, sampling_rate = torchaudio.load("/usr/users/siapartnerscomsportif/bohin_ant/MP3/input/exemple_ilyess.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)
# Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
emotion = [1, 0.05, 0.05, 0.05, 0.60, 0.60, 0.35, 1]

commentary = "Tottenham with the ball! He dribbles past a Blue defender—a powerful shot! The keeper dives—it hits the post! Rebound! Quick white shirt passes—SHOT! GOAL! Tottenham take the lead!" 

cond_dict = make_cond_dict(
    text=commentary,
    language="en-us",
    speaker=speaker,  # ou None si vous laissez le modèle générer un embedding
    emotion=emotion,
    pitch_std=100.0,
    speaking_rate=13.0,
    speaker_noised=False,
)


conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("/usr/users/siapartnerscomsportif/bohin_ant/MP3/output/test_zonos.wav", wavs[0], model.autoencoder.sampling_rate)