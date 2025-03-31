import os
import shutil
import sys
import torch
import torchaudio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(SRC_DIR)
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import gc


def generate_audio_commentary(video_path: str, commentary: str, action: str):
    """
    Generates an audio commentary from text and saves it in the `commentary/` folder next to the video.

    Args:
        video_path (str): The full path to the video file (e.g., "data/clips/clip_0_Corner.mp4").
        commentary (str): The text commentary to be converted into audio.

    Returns:
        str: Path to the generated audio file.
    """
    # Extract directory and filename from the video path
    clip_folder = os.path.dirname(video_path)  # Extracts "data/clips/"
    video_filename = os.path.basename(video_path)  # Extracts "clip_0_Corner.mp4"

    # Define the commentary folder inside the clip folder
    commentary_folder = os.path.join(clip_folder, "commentary")
    os.makedirs(commentary_folder, exist_ok=True)  # Ensure it exists

    # Define the audio filename (same as video but with .wav extension)
    audio_filename = os.path.splitext(video_filename)[0] + ".wav"
    audio_output_path = os.path.join(commentary_folder, audio_filename)

    # ESPEAK PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path where sample.py is located
    LOCAL_DIR = os.path.join(BASE_DIR, "local/bin")  # Path to espeak binaries
    LIB_DIR = os.path.join(BASE_DIR, "local/lib")  # Path to espeak libraries
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/MP3/"))

    os.environ["PATH"] = LOCAL_DIR + ":" + os.environ.get("PATH", "")
    os.environ["ESPEAK_PATH"] = os.path.join(LOCAL_DIR, "espeak")
    os.environ["PHONEMIZER_ESPEAK_PATH"] = os.environ["ESPEAK_PATH"]
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(LIB_DIR, "libespeak-ng.so")
    os.environ["ESPEAK_DATA_PATH"] = os.path.join(BASE_DIR, "local/share/espeak-ng-data")

    # Generate a default speaker embedding
    torch.manual_seed(421)  # Ensure reproducibility
    
    if action in ['Goal', "Shots on target", "Shots off target", "Penalty", "Corner", "Direct free-kick"]:
        print("ok")
        # Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
        emotion = [0.65, 0, 0, 0, 0.2, 0.3, 0.05, 1]  
        input_choice = "Short.mp3"
        speaking_rate = 17
        pitch_std = 17
    #elif action in ["Substitution", "Offside", "Kick-off", "Ball out of play", "Clearance", "Throw-in"]:
    else:
        emotion = [0.6, 0, 0, 0, 0.1, 0.3, 0.05, 1]
        input_choice = "calm_peter.mp3"
        speaking_rate = 17
        pitch_std = 10

    # Load the Zonos model
    with torch.no_grad():
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        input_audio_path = os.path.join(DATA_DIR, "input", input_choice)
        wav, sampling_rate = torchaudio.load(input_audio_path)
        speaker = model.make_speaker_embedding(wav, sampling_rate)

        cond_dict = make_cond_dict(
            text=commentary,
            language="en-us",
            speaker=speaker,  # No predefined speaker embedding
            emotion=emotion,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            speaker_noised=False,
        )

        conditioning = model.prepare_conditioning(cond_dict)
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

        # Save the generated audio
        torchaudio.save(audio_output_path, wavs[0], model.autoencoder.sampling_rate)
    del model, commentary, speaker, emotion, pitch_std, speaking_rate
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Generated audio commentary saved at: {audio_output_path}")
    return audio_output_path


if __name__ == "__main__":
    #clips = [{'path': 'data/clips/clip_0_Corner.mp4', 'label': 'Corner', 'gameTime': '1 - 00:37'}, {'path': 'data/clips/clip_2_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:34'}]
    clips = [{'path': 'data/clips/clip_0_Corner.mp4', 'label': 'Corner', 'gameTime': '1 - 00:37'}, {'path': 'data/clips/clip_4_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 02:07'}]
    #descriptions[clip["path"]] = {"timestamp": clip["gameTime"], "label": clip["label"], "description": description}
    descriptions= {'data/clips/clip_0_Corner.mp4':  {'timestamp': '1 - 00:37', 'label': 'Corner', 'description': "The Tottenham player takes a corner... the ball is whipped in... it's met by a Chelsea defender... but the Tottenham player finds space... and he fires a powerful shot... it's in! The ball sails past the keeper... it's a stunning goal!"},
                    #'data/clips/clip_4_Goal.mp4': {'timestamp':'1 - 02:07', 'label': 'Goal', 'description': "The Tottenham player breaks through the defense... he takes a shot—GOOOAL! The Chelsea goalkeeper dives but can't stop it! What a stunning finish!"}
                    'data/clips/clip_4_Goal.mp4': {'timestamp':'1 - 02:07', 'label': 'Goal', 'description': "The red player receives the ball! Defender closes in—a quick pass! The forward moves towards the box—a powerful shot! The goalkeeper dives—NO! GOOOOAAALLL!!! The crowd erupts!"}
                    }
    for clip in clips:
        clip_path= clip["path"]
        generate_audio_commentary(clip_path, descriptions[clip_path]["description"], clip['label'])


"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path of sample.py
    SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Path to src/
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))  # Root of the project

    LOCAL_DIR = os.path.join(BASE_DIR, "local/bin")  # Path to espeak binaries
    LIB_DIR = os.path.join(BASE_DIR, "local/lib")  # Path to espeak library
    DATA_DIR = os.path.join(PROJECT_ROOT, "data/MP3/")  # Path to MP3 data folder

    sys.path.append(SRC_DIR)

    os.environ["PATH"] = LOCAL_DIR + ":" + os.environ.get("PATH", "")
    os.environ["ESPEAK_PATH"] = os.path.join(LOCAL_DIR, "espeak")
    os.environ["PHONEMIZER_ESPEAK_PATH"] = os.environ["ESPEAK_PATH"]
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(LIB_DIR, "libespeak-ng.so")
    os.environ["ESPEAK_DATA_PATH"] = os.path.join(BASE_DIR, "local/share/espeak-ng-data")

    # Import modules from the correct location
    import torch
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as device

    # Load pre-trained Zonos model
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

    # Load audio file from the correct data directory
    input_audio_path = os.path.join(DATA_DIR, "input", "exemple_ilyess.mp3")
    output_audio_path = os.path.join(DATA_DIR, "output", "test_zonos_.wav")

    wav, sampling_rate = torchaudio.load(input_audio_path)
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    torch.manual_seed(421)
    # Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
    emotion = [1, 0.05, 0.05, 0.05, 0.8, 0.5, 0.5, 0.05]

    commentary = "Tottenham with the ball! He dribbles past a Blue defender—a powerful shot! The keeper dives—it hits the post! Rebound! Quick white shirt passes—SHOT! GOAL! Tottenham take the lead!" 

    cond_dict = make_cond_dict(
        text=commentary,
        language="en-us",
        speaker=speaker,  # or None if letting the model generate an embedding
        emotion=emotion,
        pitch_std=100.0,
        speaking_rate=13.0,
        speaker_noised=False,
    )

    conditioning = model.prepare_conditioning(cond_dict)

    codes = model.generate(conditioning)

    wavs = model.autoencoder.decode(codes).cpu()
    torchaudio.save(output_audio_path, wavs[0], model.autoencoder.sampling_rate)"""

