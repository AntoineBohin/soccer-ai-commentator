import os
import cv2
import subprocess
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import gc

def get_key_actions(game_folder, confidence_threshold=0.9):
    """
    Extracts key actions that have a confidence score above the threshold, sorted in chronological order.

    Args:
    - game_folder (str): The folder containing the results JSON.
    - confidence_threshold (float): Minimum confidence score to consider.

    Returns:
    - List of dictionaries containing selected actions.
    """
    json_path = os.path.join("data", "app_test", "action", game_folder, "results_spotting.json")

    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter actions with confidence > threshold and sort by frame position (chronological order)
    actions = [
        action for action in data["predictions"] if float(action["confidence"]) >= confidence_threshold
    ]
    actions.sort(key=lambda x: int(x["position"]))  # Sort chronologically by frame position
    return actions

def extract_video_clips(video_path, actions, output_folder="data/clips/", clip_length=10):
    """
    Extracts video clips around detected actions.

    Args:
    - video_path (str): Path to the input video.
    - actions (list): List of key actions with `position` (frame).
    - output_folder (str): Where to save extracted clips.
    - clip_length (int): Total clip duration in seconds.

    Returns:
    - List of extracted clip paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of the video
    cap.release()

    clips = []

    for i, action in enumerate(actions):
        action_time = int(action["position"]) / 1000  # Convert ms to seconds
        start_time = max(0, action_time - clip_length // 2)  # 5 seconds before
        end_time = action_time + clip_length // 2  # 5-10 seconds after

        # Ensure output format is always MP4
        output_clip_path = os.path.join(output_folder, f"clip_{i}_{action['label']}.mp4")

        # Extract video clip with FFmpeg, forcing MP4 output
        cmd = f"ffmpeg -i {video_path} -map 0:v -c:v libx264 -preset fast -crf 23 -ss {start_time:.2f} -to {end_time:.2f} -y {output_clip_path}"
        subprocess.run(cmd, shell=True)

        clips.append({"path": output_clip_path, "label": action["label"], "gameTime": action["gameTime"]})

    return clips

def describe_actions(clips, clip_length=10, use_tracking=True):
    descriptions = {}

    for clip in clips:
        with torch.no_grad():
            model_id = "Qwen2.5_7B_weights"
            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to('cuda')
            
            if use_tracking:
                action_context =f"""You are an expert video analyst for soccer matches.
                                We have identified an important key action in the game, classified as: **{clip['label']}**. You will analyze a **{clip_length}-second video clip (25 FPS)** that captures this action.
                                Your goal is to provide a **highly detailed, chronological, and timestamped description** of what happens during the clip so that a language model can later generate a rich and accurate spoken commentary.

                                ---

                                **Important Instructions:**
                                - **Use the full set of frames provided** (≈40 frames sampled at 4 FPS).
                                Group your analysis into 1-second chunks (4 frames = 1 second). Use headings like: **Time: 0s–1s**, **1s–2s**, etc.
                                Within each section, summarize what happens in those frames — DO NOT repeat the same description.

                                - Within each time block, describe the **main movements and interactions** during that segment.
                                - Focus on **player actions**, **ball trajectory**, **interactions**, **team roles** (based on jersey color), and **goal attempts**.
                                - Use information from visual cues:
                                    - Players = **orange**, Referees = **yellow**, Goalkeepers = **blue**, Ball = **purple** (⚠️ not always accurate).
                                - ⚠️ **Avoid repeating the same shot/save multiple times.** If an action spans 2–3 frames, **describe it only once**.
                                - Be vivid but do **not hallucinate** — describe what is visible, frame-by-frame.
                                """
                # action_context =f"""You are an expert video analyst for soccer matches.
                #                 We have identified an important key action in the game, classified as: **{clip['label']}**. You will analyze a **{clip_length}-second video clip (25 FPS)** that captures this action.
                #                 Your goal is to provide a **highly detailed, chronological, and timestamped description** of what happens during the clip so that a language model can later generate a rich and accurate spoken commentary.

                #                 **Important Instructions:**
                #                 - **Use the full set of frames provided** (≈40 frames sampled at 4 FPS).
                #                 Group your analysis into 5-seconds chunks (20 frames = 5 seconds). Use these 2 headings only since you have a 10 seconds video : **Time: 0s–5s**, **5s–10s**
                #                 Within each section, summarize what happens in those frames.
                #                 - Within each time block, describe the **main movements and interactions** during that segment.
                #                 """

            else:
                action_context =f"""You are an expert video analyst for soccer matches. 
                            We have identified an important action classified as: **{clip['label']}**. Use it to understand the action.
                            You will analyze a **{clip_length}-SECOND** video clip capturing this action.

                            Be careful: Your analysis must be **CHRONOLOGICAL - DESCRIBE THE ACTION STEP-BY-STEP WITH TIMESTAMPS (betwen 0 and {clip_length} sec)**, from the beginning of the clip to the end. 
                            
                            Your role is to describe the action STEP BY STEP EXHAUSTIVELY:
                            - Identify each player's **MOVEMENTS** and **POSITIONING**.
                            - Track the **BALL'S TRAJECTORY** and **HOW IT IS PASSED, INTERCEPTED or SHOT**.
                            - You can describe **WHICH TEAM** has the ball or defends using the **JERSEY COLORS**.
                            - Detail **INTERACTIONS BETWEEN PLAYERS** (e.g., duels, passes, pressing).
                            - Include relevant **CONTEXTUAL ELEMENTS** (e.g., referee’s decisions, reactions from teammates or opponents).
                            
                            ENSURE THAT YOUR DESCRIPTION IS HIGHLY DETAILED AND STRUCTURED,** so it provides enough information for an **LLM TO LATER GENERATE A FULL AND NATURAL SPORTS COMMENTARY** of this action.
                            **AVOID SUMMARIZATION**—focus on **PRECISE DETAILS** to ensure a **VIVID AND ACCURATE** account of the scene.
                            """

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": clip["path"],
                        #"max_pixels": 270 * 315,
                        "max_pixels": 60000,
                        "fps": 4.0  
                    },
                    {
                        "type": "text",
                        "text": action_context
                    }
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            print("🎞️ Number of frames processed:", video_inputs[0].shape[0], f"label = {clip['label']}")
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            generated_ids = model.generate(**inputs, max_new_tokens=1028) #do_sample=True, top_p=0.9)
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:] 
            description = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]


            print(description)

            descriptions[clip["path"]] = {"timestamp": clip["gameTime"], "label": clip["label"], "description": description}
        del processor, model, messages, text, inputs, image_inputs, video_inputs, generated_ids, generated_ids_trimmed, description
        torch.cuda.empty_cache()
        gc.collect()

    return descriptions


if __name__ == "__main__":
    folder_name, video_name = "custom", "1_720p.mkv"
    experiment, split = "action_sampling_weights_002", "test"
    game = folder_name + "/" + video_name
    prediction_dir = os.path.join("data", "app_test", "action/")

    original_clips = [{'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_0_Corner.mp4', 'label': 'Corner', 'gameTime': '1 - 00:37'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_4_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:34'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_6_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 01:52'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_7_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 02:07'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_8_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 02:11'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_9_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 03:40'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_10_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 04:31'}, 
        {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_12_Throw-in.mp4', 'label': 'Throw-in', 'gameTime': '1 - 06:21'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_13_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:33'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_14_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 06:41'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_15_Goal.mp4', 'label': 'Goal', 'gameTime': '1 - 07:48'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_16_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:44'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_17_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 08:46'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_18_Kick-off.mp4', 'label': 'Kick-off', 'gameTime': '1 - 09:17'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_20_Clearance.mp4', 'label': 'Clearance', 'gameTime': '1 - 10:05'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_21_Foul.mp4', 'label': 'Foul', 'gameTime': '1 - 11:00'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_22_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:06'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_23_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:08'}, {'path': 'data/app_test/action/custom/1_720p.mkv/clips/clip_24_Substitution.mp4', 'label': 'Substitution', 'gameTime': '1 - 11:39'}]

    clips = [{"path": clip["path"].replace("clips", "clips_tracked"), "label": clip["label"], "gameTime": clip["gameTime"]} for clip in original_clips]

    # Generate descriptions
    descriptions = describe_actions(clips, clip_length=10, use_tracking=True)
    print(descriptions)

    #yield "✅ Descriptions generated! Displaying results..."





action_context ="""You are a professional live sports commentator. We have identified an important action classified as: **{clip['label']}**. Use it to understand the action.
                        You will analyze a **{clip_length}-SECOND** video clip capturing this action.
                        Your job is to generate a natural, emotionally engaging **live soccer commentary** based on the given video clip.

                        **Important Guidelines to analyse the action**:
                        - Identify each player's **MOVEMENTS** and **POSITIONING**.
                        - Track the **BALL'S TRAJECTORY** and **HOW IT IS PASSED, INTERCEPTED or SHOT**.
                        - You can describe **WHICH TEAM** has the ball or defends using the **JERSEY COLORS** and the **TEAM NAMES** accessible in the MATCH CONTEXT: {match_context}.
                        - Detail **INTERACTIONS BETWEEN PLAYERS** (e.g., duels, passes, pressing).
                        - Include relevant **CONTEXTUAL ELEMENTS** (e.g., referee’s decisions, reactions from teammates or opponents).
                        

                        **Important Guidelines for the commentaries**:
                        - **IMAGINE YOU ARE LIVE ON AIR** describing this moment in real time.
                        - Your commentary will be **converted into an audio track using a text-to-speech model**.
                        - Your output must be **AS NATURAL, EMOTIONAL, and HUMAN as POSSIBLE**, with the **RIGHT LEVEL OF EXCITEMENT**.
                        - **Your commentary must fit exactly within 10 seconds**, so **BE CONCISE YET DESCRIPTIVE**.
                        - **Follow the exact chronology of the action**, as if you were watching and reacting to it live.
                        - Use **VERY SHORT AND ENERGETIC SENTENCES**, avoid unnecessary introductions.
                        - **Generate ONLY ONE version** of the commentary—do NOT provide multiple variations or repeat the same event.
                        - **DO NOT include extra labels like "Commentary:" or explanations about what you are doing.**
                        - **DO NOT REPEAT THE TIMESTAMPS** you see in the description.
                        - **USE THE MATCH CONTEXT and the jersey-to-team association** to name the players (e.g., "the Tottenham player").
                        - **IF a described jersey color does NOT match the match context, IGNORE it** and use the correct team.
                        - **Cover the entire action sequence within 10 seconds, ensuring a fluid progression from start to finish.**
                        - **ONCE THE COMMENTARY IS COMPLETE, STOP GENERATING IMMEDIATELY.**

                        **Example Style (for a penalty goal)**:
                        - "The Tottenham player steps up... he shoots—GOOOAL! The keeper dives the wrong way! What a finish!"
                        - "Oh, what a pass! The striker controls it—LEFT FOOT SHOT—AND IT’S IN!"
                        - "The ball is crossed in... header! It bounces—THE GOALKEEPER CAN'T REACH IT! Incredible!"

                        **Now, generate EXACTLY ONE 20-second commentary using the given video clip:**  
                        """


"""
Potential prompts:

Pay close attention to possible **CHANGES IN CAMERA ANGLES OR PERSPECTIVES**, but  **DON'T DESCRIBE THOS CAMERA CHANGES**.

We have detected a key soccer action that is classified as: {clip[‘label’]}. You will receive a 10-second video clip centered around this action.
Your task is to analyze and describe the sequence of movements in as much detail as possible, focusing on players’ positioning, gestures, ball trajectory, and interactions between players.
Do not summarize—describe each movement step by step. Your description will be used as raw input for a sports commentary generation model.

You are an expert video analyst for soccer matches. We have identified an important action classified as: {clip[‘label’]}.
You will analyze a 10-second video clip capturing this action. Be careful: Your analysis must be chronological—describe the action step by step, from the beginning of the clip to the end.
Your role is to describe the action exhaustively: identify each player’s movements, the ball’s trajectory and how it is passed, intercepted, or shot, interactions between players, tactical positioning, and any relevant contextual elements (e.g., referee’s position, reactions from teammates or opponents).
Pay close attention to possible changes in camera angles or perspectives.
Describe the scene in a structured and detailed manner to provide a precise account of what happens. Ensure that the description provides enough information for an LLM to later generate a full/natural sports commentary of this action. Avoid summaries—focus on precise details to ensure a vivid and accurate account of the scene.

We have detected a key action in a soccer match, identified as: {clip[‘label’]}.
We are providing you with a 10-second clip surrounding this action.
Your task is to act as a detailed observer: analyze every movement in the play. What happens before, during, and after the key action? How do players position themselves? How does the ball move? How does the action unfold from a tactical perspective?
Describe everything in precise detail, ensuring no element is overlooked. Your description will later be refined into a sports broadcast-style commentary.

We have detected a soccer key action labeled as: {clip[‘label’]}.
You are analyzing a 10-second video clip centered around this action.
Describe everything that happens in detail: player movements, ball trajectory, tactical elements, and any significant reactions.
Pay attention to what is visible in the frame: which players are in focus, any background elements that are relevant (referee, bench reactions), and the evolution of the play from a visual perspective.
Your description should be exhaustive and structured, ensuring it provides enough information for an LLM to later generate a full sports commentary.
"""

"""
RESULTS:
{'1 - 01:34': "### Video Clip Analysis: Throw-in\n\n#### Scene Description:\nThe video clip captures a throw-in action in a soccer match. The sequence begins with the ball being thrown into play by one of the players. The camera angle is positioned at a medium distance, providing a clear view of the throw-in area and the immediate vicinity around it. The field is well-lit, suggesting that the match is taking place during the day under stadium lights.\n\n#### Step-by-Step Analysis:\n\n1. **Initial Setup:**\n   - The ball is stationary on the ground near the corner flag.\n   - A player from the attacking team (let's call them Team A) is preparing to take the throw-in. This player is standing outside the penalty area, holding the ball with both hands behind their head.\n   - The defending team (Team B) is positioned around the throw-in spot, ready to react to the throw-in.\n\n2. **Throw-In Execution:**\n   - The player from Team A throws the ball into the air with both hands, ensuring it crosses the sideline.\n   - The ball is thrown with a moderate force, aiming to reach the center of the field.\n   - As the ball is released, the camera follows its trajectory, maintaining focus on the ball and the immediate area around the throw-in spot.\n\n3. **Ball's Trajectory:**\n   - The ball travels through the air, curving slightly due to the thrower's technique.\n   - Upon reaching the intended target, the ball lands on the ground within the field of play.\n   - The ball's landing point is near the center of the field, close to the halfway line.\n\n4. **Interactions Between Players:**\n   - As the ball is thrown, players from both teams begin to move towards it, anticipating possession.\n   - A defender from Team B jumps to intercept the ball, but the thrower from Team A manages to control the ball before it reaches the defender.\n   - The thrower from Team A quickly passes the ball to a teammate who is positioned further up the field, likely looking to create an offensive opportunity.\n\n5. **Tactical Positioning:**\n   - The attacking team (Team A) has positioned themselves strategically to capitalize on the throw-in. The thrower is positioned outside the penalty area, while other teammates are spread out across the field, ready to receive the pass.\n   - The defending team (Team B) has positioned themselves to contest the throw-in, with defenders and midfielders forming a defensive wall around the throw-in"}
{'1 - 02:07': "### Video Clip Analysis: Goal\n\n#### Scene Description:\nThe video clip captures a critical moment in a soccer match where a goal is scored. The sequence begins with the ball being played into the penalty area by a midfielder from the attacking team. The camera angle is a wide shot that covers the entire penalty box, allowing us to see all players involved in the play.\n\n---\n\n#### Step-by-Step Analysis:\n\n1. **Initial Play (0-2 seconds):**\n   - **Midfielder (Attacking Team):** The midfielder receives the ball near the halfway line and starts dribbling towards the penalty area. His movements are quick and agile, evading defenders.\n   - **Defender (Defending Team):** A defender from the defending team attempts to intercept the ball but fails due to the midfielder's skillful dribbling.\n   - **Ball Trajectory:** The ball is initially passed forward by the midfielder, then deflected slightly to the right as he maneuvers past the defender.\n   - **Camera Angle:** The camera remains static, providing a clear view of the midfield action as the ball enters the penalty area.\n\n2. **Penalty Area Play (2-4 seconds):**\n   - **Midfielder (Attacking Team):** As the ball reaches the penalty area, the midfielder passes the ball to a forward who is positioned near the edge of the box.\n   - **Forward (Attacking Team):** The forward receives the ball and takes a few steps back before passing the ball to another attacker positioned deeper inside the box.\n   - **Defender (Defending Team):** A defender from the defending team tries to press the forward but is unable to intercept the pass.\n   - **Ball Trajectory:** The ball is passed horizontally across the penalty area, moving from one attacker to another.\n   - **Camera Angle:** The camera angle shifts slightly to follow the movement of the ball as it crosses the penalty area.\n\n3. **Final Pass (4-6 seconds):**\n   - **Attacker (Attacking Team):** The ball is received by an attacker who is positioned near the penalty spot. This attacker has a clear view of the goal and decides to take a shot.\n   - **Defender (Defending Team):** A defender from the defending team attempts to block the shot but is too late.\n   - **Ball Trajectory:** The ball is shot directly at the goal, and the goalkeeper dives to make a save but misses the ball.\n   - **Camera Angle:** The camera zooms in"}
"""


# action_context =f"""You are an expert video analyst for soccer matches.
#                 We have identified an important key action in the game, classified as: **{clip['label']}**. You will analyze a **{clip_length}-second video clip (25 FPS)** that captures this action.
#                 Your goal is to provide a **highly detailed, chronological, and timestamped description** of what happens during the clip so that a language model can later generate a rich and accurate spoken commentary.

#                 ---

#                 **Important Instructions:**
#                 1. **Use the full set of frames provided** (≈30 frames sampled at 3 FPS).
#                 Group your analysis into 1-second chunks (3 frames = 1 second). Use headings like: **Time: 0s–1s**, **1s–2s**, etc.
#                 Within each section, summarize what happens in those frames — DO NOT repeat the same description.

#                 4. **TRACKING HELP (Visual Annotations)**:
#                 - **Players** are highlighted with **orange ellipses**.
#                 - **Referees** have **yellow ellipses**.
#                 - **Goalkeepers** are marked with **blue ellipses**.
#                 - The **ball** is highlighted in **purple**, but **this tracking is often unreliable**. Use it only as a **clue**, not a definitive source.

#                 5. **Jersey Colors for Team Identification**:
#                 - Describe players using their **jersey colors** (e.g., "white jersey player passes to a dark blue jersey forward").

#                 6. **WHAT TO DESCRIBE**:
#                 - **Player movements and positioning**.
#                 - **Ball trajectory**: passes, interceptions, shots, clearances.
#                 - **Interactions**: tackles, duels, assists, defensive pressure.
#                 - **Context**: referee signals, player gestures, group reactions (celebration, disappointment, etc.).

#                 **FORMAT REQUIREMENTS**:
#                 - Be **objective, structured, and vivid**.
#                 - DO NOT summarize—**focus on each action** with **granularity** and clarity.
#                 - Avoid any speculation; base your description only on what is **visibly happening** in the video.
#                 - You can use bullet points or paragraphs, but it must follow a **timeline**.

#                 Your output will be used by another model to generate **spoken sport commentary**, so your analysis must be clear, exhaustive, and grounded in the actual content of the video.
#                 """
