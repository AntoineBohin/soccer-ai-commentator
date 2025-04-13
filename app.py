import gradio as gr
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

import io
import time
import re
import sys
import cv2
import subprocess
import pickle
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from contextlib import redirect_stdout
from pathlib import Path

from src import constants
from scripts.action.predict import predict_single_video
from src.vlm.vlm_utils import get_key_actions, extract_video_clips, describe_actions
from src.vlm.llm_utils import generate_commentary
from src.player_tracking.main import create_tracked_video

sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/zonos"))
from zonos.sample import generate_audio_commentary


VIDEO_DIRECTORY = "./data/soccernet/action-spotting-2023"


def list_folders():
    """Liste les dossiers disponibles contenant des vidéos"""
    folders = [f for f in os.listdir(VIDEO_DIRECTORY) if os.path.isdir(os.path.join(VIDEO_DIRECTORY, f))]
    return folders

def list_videos_in_folder(folder_name):
    """Liste les vidéos dans le dossier sélectionné"""
    if folder_name is None or folder_name == "Aucun dossier disponible":
        return gr.update(choices=[], value=None)  # Pas de vidéos sélectionnables

    folder_path = os.path.join(VIDEO_DIRECTORY, folder_name)

    if not os.path.exists(folder_path):  # Vérifie que le dossier existe
        return gr.update(choices=[], value=None)

    videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if videos:
        return gr.update(choices=videos, value=videos[0])  # Met la première vidéo par défaut
    else:
        return gr.update(choices=[], value=None)  # Aucune vidéo trouvée

def toggle_video_dropdown(mode_choice):
    if mode_choice == "Analyze specific extract":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def merge_video_with_audio(video_path: str, audio_path: str, output_path: str):
    """
    Merges a video (.mp4) with an audio commentary (.mp3) and saves the result.
    If the original video has no audio stream, it first adds a silent track.

    Args:
        video_path (str): Path to the original video file.
        audio_path (str): Path to the corresponding audio commentary (.mp3).
        output_path (str): Path where the final video will be saved.

    Returns:
        str: Path to the saved merged video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output folder exists

    # If the video has no audio, add a silent track
    merge_cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i", video_path,  # Input video
        "-i", audio_path,  # Input audio
        #"-c:a", "aac",
        "-acodec", "copy",
        "-vcodec", "copy",
        "-c:v", "libx264",        # Re-encode video to H.264
        "-c:a", "aac",             # Re-encode audio to AAC
        "-shortest",
        output_path
    ]

    try:
        subprocess.run(merge_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Merged: {video_path} + {audio_path} → {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error merging {video_path} and {audio_path}:\n{e.stderr.decode()}")
        return None


def convert_wav_to_mp3(input_wav: str, output_mp3: str, bitrate="192k"):
    """
    Converts a WAV file to MP3 using FFmpeg.
    Args:
        input_wav (str): Path to the input WAV file.
        output_mp3 (str): Path to the output MP3 file.
        bitrate (str, optional): Bitrate for the MP3 file (default: 192k).
    Returns:
        str: Path to the converted MP3 file, or None if an error occurred.
    """
    command = [
        "ffmpeg",
        "-i", input_wav,  # Input WAV file
        "-c:a", "libmp3lame",  # Use LAME MP3 encoder
        "-b:a", bitrate,  # Set audio bitrate
        output_mp3  # Output MP3 file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted: {input_wav} → {output_mp3}")
        return output_mp3
    except subprocess.CalledProcessError as e:
        print(f" Error converting {input_wav} to {output_mp3}:\n{e.stderr.decode()}")
        return None


def merge_all_videos_with_audio(CLIPS_DIR, COMMENTARY_DIR, OUTPUT_DIR):
    """
    Merges all videos in `data/clips/` with their corresponding audio commentaries in `data/clips/commentary/`
    and saves them in `data/commented_clips/`.
    """
    # Define paths
    #BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    #CLIPS_DIR = os.path.join(BASE_DIR, "data/clips")
    #COMMENTARY_DIR = os.path.join(CLIPS_DIR, "commentary")
    #OUTPUT_DIR = os.path.join(BASE_DIR, "data/commented_clips")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output folder exists

    # Convert all WAV files to MP3
    for audio_file in os.listdir(COMMENTARY_DIR):
        if audio_file.endswith(".wav"):
            wav_path = os.path.join(COMMENTARY_DIR, audio_file)
            mp3_path = os.path.join(COMMENTARY_DIR, os.path.splitext(audio_file)[0] + ".mp3")
            if not os.path.exists(mp3_path):  # Convert only if MP3 doesn't exist
                convert_wav_to_mp3(wav_path, mp3_path)

    # Process each video in `data/clips/`
    for video_file in os.listdir(CLIPS_DIR):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(CLIPS_DIR, video_file)
            audio_path = os.path.join(COMMENTARY_DIR, os.path.splitext(video_file)[0] + ".mp3")
            output_path = os.path.join(OUTPUT_DIR, video_file) #video_file.split(".")[0]+".mkv")  # Save with same name but in `commented_clips/`

            # Ensure the corresponding audio exists
            if os.path.exists(audio_path):
                merge_video_with_audio(video_path, audio_path, output_path)
            else:
                print(f"⚠ No matching audio found for {video_file}. Skipping.")
    

def full_pipeline(folder_name, video_name, video_prediction=True, clip_extraction=True, use_tracking=True, generate_description=True, gen_commentary=True, generate_audio=True, single_match=False):
    
    experiment, split = "action_sampling_weights_002", "test"
    ############## INFO ABOUT THE VIDEO ##############
    if single_match:
        game = folder_name + "/" + video_name
    else:
        game = folder_name
    video = cv2.VideoCapture(VIDEO_DIRECTORY + "/" + folder_name + "/" + video_name)
    constants.video_fps = float(video.get(cv2.CAP_PROP_FPS))

    constants.test_games = [game]
    prediction_dir = os.path.join("data", "app_test", "action/")

    clips_dir=os.path.join(prediction_dir, game, "clips")
    output_dir=os.path.join(prediction_dir, game, "clips_commented")

    clip_cache_path = os.path.join(prediction_dir, game, "cache", "original_clips.pkl")
    os.makedirs(os.path.dirname(clip_cache_path), exist_ok=True)

    desc_cache_path = os.path.join(prediction_dir, game, "cache", "descriptions.pkl")
    os.makedirs(os.path.dirname(desc_cache_path), exist_ok=True)

    commentary_cache_path = os.path.join(prediction_dir, game, "cache", "commentary.pkl")
    os.makedirs(os.path.dirname(commentary_cache_path), exist_ok=True)

    ######################### ACTION SPOTTING #############################
    yield "Running Action Spotting model on video...", [], {}, {}, {}
    if video_prediction:
        predict_single_video(experiment, split, 0, False, Path(prediction_dir), constants, single_match=False)
    else:
        time.sleep(2)

    ########################## CLIP EXTRACTION ############################
    actions = get_key_actions(game, confidence_threshold=0.8)

    yield f"Action Spotting completed, {len(actions)} key actions were identified! Extraction of the highlight clips...", [], {}, {}, {}
    if clip_extraction:
        original_clips = extract_video_clips(os.path.join(VIDEO_DIRECTORY, game), actions, clips_dir, clip_length=10)
        with open(clip_cache_path, "wb") as f:
            pickle.dump(original_clips, f)
    else:
        if os.path.exists(clip_cache_path):
            with open(clip_cache_path, "rb") as f:
                original_clips = pickle.load(f)
        else:
            raise FileNotFoundError(f"Commentary cache not found at {commentary_cache_path}. Run with gen_commentary=True first.")
        time.sleep(2)
    original_clips = []
    for i, action in enumerate(actions):
        label = action["label"].replace(" ", "_")
        output_clip_path = os.path.join(clips_dir, f"clip_{i}_{label}.mp4")
        original_clips.append({"path": output_clip_path, "label": action["label"], "gameTime": action["gameTime"]})

    ######################## CLIP MAPPING ################################
    if use_tracking:
        os.makedirs(os.path.join(prediction_dir, game,"clips_tracked"), exist_ok=True)
        # Create a tracked video for each clip
        for clip in original_clips:
            clip_path = clip["path"]
            clip_path = clip_path.replace(" ", "_")
            video_path = clip['path'].replace("clips", "clips_tracked")
            video_path = video_path.replace(" ", "_")
            create_tracked_video(
                source_video_path=clip_path,
                target_video_path=video_path,
                device="cuda"
            )
        clips = [{"path": clip["path"].replace("clips", "clips_tracked"), "label": clip["label"], "gameTime": clip["gameTime"]} for clip in original_clips]
        commentary_dir=os.path.join(prediction_dir, game, "clips_tracked", "commentary")
    else:
        clips = original_clips
        commentary_dir=os.path.join(prediction_dir, game, "clips", "commentary")
    clips = [{"path": clip["path"].replace("clips", "clips_tracked"), "label": clip["label"], "gameTime": clip["gameTime"]} for clip in original_clips]
    commentary_dir=os.path.join(prediction_dir, game, "clips_tracked", "commentary")

    clip_choices = [f"{clip['gameTime']} {clip['label']}" for clip in original_clips]
    clip_mapping = {f"{clip['gameTime']} {clip['label']}": clip["path"].replace("clips", "clips_commented") for clip in original_clips}
    yield f"Highlight clips extracted! Generating AI descriptions...", gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), {}, {}, {}
    
    ###################### DESCRIPTION GENERATION #######################
    # match_context = "It's a Premier League match between Tottenham and Chelsea. Tottenham plays with white jerseys, and Chelsea plays with dark blue jerseys."
    # match_context = "It's a futsal game between two teams, blue jersey vs green jersey. The game is played in a small indoor court with a goal at each end."
    match_context = "It's a football game between two teams wearing differents jerseys where we have to comment the actions of the game. PSG is playing in white jerseys, Barcelona in blue with red stripes jersey."
    if generate_description:
        descriptions = describe_actions(clips, clip_length=10, use_tracking=True)
        with open(desc_cache_path, "wb") as f:
            pickle.dump(descriptions, f)
    else:
        if os.path.exists(desc_cache_path):
            with open(desc_cache_path, "rb") as f:
                descriptions = pickle.load(f)
        else:
            raise FileNotFoundError(f"Description cache not found at {desc_cache_path}. Run with generate_description=True first.")
        time.sleep(2)
    
    # descriptions dictionary has the format: descriptions[clip["path"]] = {"timestamp": clip["gameTime"], "label": clip["label"], "description": description}
    yield f"AI Descriptions generated! Now we reformulate them into natural sport commentaries...", gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), {}, {}, {}
    ################### TEXTUAL COMMENTARY ###########################
    if gen_commentary:
        results = generate_commentary(descriptions, match_context)
        with open(commentary_cache_path, "wb") as f:
            pickle.dump(results, f)
    else:
        if os.path.exists(commentary_cache_path):
            with open(commentary_cache_path, "rb") as f:
                results = pickle.load(f)
        time.sleep(2)

    yield f"Textual commentaries generated! Now generating audio commentaries and merging with the video clips...", gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None, visible=True), {}, {}, {}
    if generate_audio:
        for clip in clips:
            clip_path= clip["path"]
            generate_audio_commentary(clip_path, results[clip_path]["Generated Commentary"], clip['label'])
        merge_all_videos_with_audio(CLIPS_DIR=clips_dir, COMMENTARY_DIR=commentary_dir, OUTPUT_DIR=output_dir)
    else:   
        time.sleep(2)

    ############ Extract list of unique labels for filtering the highlights ###########
    unique_labels = sorted(set([clip["label"] for clip in original_clips]))
    label_to_highlights = {}
    for label in unique_labels:
        label_to_highlights[label] = [
            f"{clip['gameTime']} {clip['label']}"
            for clip in original_clips if clip["label"] == label
        ]
        
    yield "✅ Generation complete! Select a highlight to view.", gr.update(choices=['Choose a highlight'] + clip_choices, value='Choose a highlight', visible=True), clip_mapping, results, descriptions, unique_labels, label_to_highlights, gr.update(choices=unique_labels, value=unique_labels[0] if unique_labels else None)

def run_pipeline_with_mode(folder, video, mode,
                           video_prediction, clip_extraction, use_tracking,
                           generate_description, gen_commentary, generate_audio):
    single_match = (mode == "Analyze specific extract")
    
    # Consomme le générateur pour récupérer le dernier yield
    pipeline = full_pipeline(
        folder_name=folder,
        video_name=video,
        video_prediction=video_prediction,
        clip_extraction=clip_extraction,
        use_tracking=use_tracking,
        generate_description=generate_description,
        gen_commentary=gen_commentary,
        generate_audio=generate_audio,
        single_match=single_match
    )

    final_output = None
    for output in pipeline:
        final_output = output  # keep updating until we get the last one
    
    return final_output

def display_highlight(clip_name, clip_mapping, results, descriptions):
    """Updates video display and text when a highlight is selected."""
    if not clip_name or (isinstance(clip_name, list) and len(clip_name) == 0):
        # Return only two outputs: video (None) and an error message.
        return None, None, "⚠️ Please select a highlight.", "⚠️ Please select a highlight."
    
    # Ensure `clip_name` is a string
    if isinstance(clip_name, list):
        clip_name = clip_name[0]
    if not isinstance(clip_name, str):
        return None, None, "⚠️ Invalid selection.", "⚠️ Please select a highlight."
    
    # Ensure `clip_mapping` is a dictionary
    if not isinstance(clip_mapping, dict):
        return None, None, "⚠️ Error: Clip mapping is missing.", "⚠️ Please select a highlight."
    
    # Retrieve the correct video path
    if clip_name in clip_mapping:
        video_path = clip_mapping[clip_name]
        video_path = video_path.replace("clips_tracked", "clips_commented")
        result_path = video_path.replace("clips_commented", "clips_tracked")
        print(f"video path : {video_path}, result path {result_path}")
        result = results.get(result_path, {}).get("Generated Commentary", "ℹ️ No description available.")
        description = descriptions[result_path]['description']
        return result_path, video_path, result, description 

    return None, None, "❌ Selected highlight not found.", "❌ Selected highlight not found."

def update_highlight_dropdown(selected_label, label_to_highlights):
    if not selected_label or selected_label not in label_to_highlights:
        return gr.update(choices=["No available highlight"], value=None)
    choices = label_to_highlights[selected_label]
    return gr.update(choices=choices, value=choices[0] if choices else None)


original_clips = [{'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_0_Corner.mp4', 'label': 'Corner', 'gameTime': '1 - 00:37'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_4_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:34'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_6_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:52'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_7_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 02:07'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_8_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 02:11'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_9_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 03:40'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_10_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 04:31'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_12_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 06:21'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_13_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:33'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_14_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:41'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_15_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 07:48'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_16_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:44'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_17_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:46'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_18_Kick-off.mp4', 'label': 'Kick-off', 'gameTime': '1 - 09:17'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_20_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 10:05'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_21_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 11:00'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_22_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:06'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_23_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:08'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_24_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:39'}]


# Callback pour changer la visibilité des zones vidéo en fonction de la sélection de la source
def update_video_source(source):
    if source == "Tracked Video":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks() as demo:
    gr.Markdown("# 🎥 AI-Powered Soccer Match Commentary")

    folder_dropdown = gr.Dropdown(choices=list_folders(), label="📂 Select a folder")

    #################### CHOSE THE PARAMETERS ######################
    mode_radio = gr.Radio(
        choices=["Analyze specific extract", "Analyze full match (2 halves)"],
        label="🎯 Choose Analysis Mode",
        value="Analyze specific extract"
    )
    gr.Markdown("Check the following boxes if you want to run the different steps or use the cached results to reduce inference time.")
    video_prediction_checkbox = gr.Checkbox(label="🎯 Run Action Spotting", value=True)
    clip_extraction_checkbox = gr.Checkbox(label="🎬 Run Clips Extraction", value=True)
    use_tracking_checkbox = gr.Checkbox(label="📌 Use Player Tracking", value=True)
    generate_description_checkbox = gr.Checkbox(label="📝 Descriptions Generation", value=True)
    gen_commentary_checkbox = gr.Checkbox(label="🗣️ Commentary Generation", value=True)
    generate_audio_checkbox = gr.Checkbox(label="🔊 Audio Generation", value=True)

    ########################### VIDEO SELECTION IF NEEDED ############################
    video_dropdown = gr.Dropdown(label="🎥 Select a soccer match", choices=["No .mkv files available"]) #"Sélectionnez d'abord un dossier"
    mode_radio.change(fn=toggle_video_dropdown, inputs=mode_radio, outputs=video_dropdown)
    folder_dropdown.change(fn=list_videos_in_folder, inputs=folder_dropdown, outputs=video_dropdown)

    ###################### START GENERATION #######################
    start_button = gr.Button("🚀 Start Commentary Generation")
    output_textbox = gr.Textbox(label="Generation Status", interactive=False)

    video_source_radio = gr.Radio(choices=["Tracked Video", "Commented Video"], label="Select a video", value="Tracked Video")

    ###################### HIGHLIGHT SELECTION ########################
    action_type_dropdown = gr.Dropdown(label="🏷️ Select Action Type", choices=["No types yet"], value=None, visible=True)
    highlights_dropdown = gr.Dropdown(label="📌 Select a Highlight", choices=["No available highlight"], visible=False)# Zones vidéo
    video_player_tracked = gr.Video(label="🎬 Tracked Video", visible=True)
    video_player_commented = gr.Video(label="🎬 Commented Video", visible=False)
    # video_player = gr.Video(label="🎬 Highlight Clip", visible=False)
    description_textbox = gr.Textbox(label="📝 VLM Output Description", visible=True)
    result_textbox = gr.Textbox(label="📝 AI-Generated Commentary", visible=False)

    stored_clip_mapping = gr.State({})  
    stored_results = gr.State({})
    stored_description = gr.State({})
    stored_action_labels = gr.State([])  # ⬅️ Ajoute ces 2 états
    stored_action_to_highlights = gr.State({})

    start_button.click(
        fn=run_pipeline_with_mode,
        inputs=[
            folder_dropdown, video_dropdown, mode_radio,
            video_prediction_checkbox, clip_extraction_checkbox, use_tracking_checkbox,
            generate_description_checkbox, gen_commentary_checkbox, generate_audio_checkbox
        ],
        outputs=[
            output_textbox, highlights_dropdown, stored_clip_mapping,
            stored_results, stored_description, stored_action_labels, stored_action_to_highlights, action_type_dropdown
        ]
    )

    ###################### FILTER HIGHLIGHTS BY LABEL ########################
    action_type_dropdown.change(
        fn=update_highlight_dropdown,
        inputs=[action_type_dropdown, stored_action_to_highlights],
        outputs=highlights_dropdown
    )

    ###################### DISPLAY HIGHLIGHT ########################
    highlights_dropdown.change(
        fn=display_highlight,
        inputs=[highlights_dropdown, stored_clip_mapping, stored_results, stored_description],
        outputs=[video_player_tracked, video_player_commented, result_textbox, description_textbox]
    )

    ###################### VIDEO TOGGLE ########################
    video_source_radio.change(
        fn=update_video_source,
        inputs=video_source_radio,
        outputs=[video_player_tracked, video_player_commented]
    )

    # highlights_dropdown.change(fn=display_highlight, inputs=[highlights_dropdown, stored_clip_mapping, stored_results, stored_description], outputs=[video_player_tracked, video_player_commented, result_textbox, description_textbox])

    # video_source_radio.change(
    #     fn=update_video_source, 
    #     inputs=video_source_radio, 
    #     outputs=[video_player_tracked, video_player_commented]
    # )

    highlights_dropdown.change(lambda: gr.update(visible=True), outputs=highlights_dropdown)
    highlights_dropdown.change(lambda: gr.update(visible=True), outputs=description_textbox)
    highlights_dropdown.change(lambda: gr.update(visible=True), outputs=result_textbox)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)  # Modifier si besoin
