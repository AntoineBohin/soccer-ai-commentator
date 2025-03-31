import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# Chargement du modèle sur GPU (utilisez "cpu" si vous n'avez pas de GPU)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")

# # Charger un fichier audio d'exemple
# wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")

# # Créer une embedding pour le speaker
# speaker = model.make_speaker_embedding(wav, sampling_rate)

# # Préparer le dictionnaire de conditionnement
# cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
# conditioning = model.prepare_conditioning(cond_dict)

# # Générer les codes internes
# codes = model.generate(conditioning)

# # Décoder ces codes pour obtenir le waveform final et sauvegarder en fichier WAV
# wavs = model.autoencoder.decode(codes).cpu()
# torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
