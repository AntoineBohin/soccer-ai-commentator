import os
#os.environ["HF_TOKEN"] 
os.environ["PYTORCH_SDP_DISABLE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
import pandas as pd
import re
import gc



class Commentator:
    def __init__(self):
        """
        Initialize the Gemma4 model for text generation.
        """
        #print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        print("Loading model (16-bit mode activated)...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it",
            torch_dtype=torch.bfloat16
            #device_map="auto"
        ).eval().to("cuda")

    def rewrite_text(self, prompt, max_length=500):
        """
        Reformulates a detailed VLM description into a concise and professional sports commentary.
        """
        print("Generating commentary...")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful sports commentator."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
        
        inputs = {
                k: (
                    v.to(self.model.device, dtype=torch.bfloat16) if v.dtype.is_floating_point
                    else v.to(self.model.device)
                )
                for k, v in inputs.items()
            }
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=True)
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print(f"Commentary: {decoded}\n")

        return decoded


def generate_commentary(descriptions, match_context):
    """
    Converts detailed VLM-generated action descriptions into natural sports commentary.

    Args:
    - descriptions (dict): Dictionary mapping game timestamps to {label, description}.
    - match_context (str): General match context (score, teams, situation).
    - num_variations (int): Number of different commentary versions per action.
    - save_path (str, optional): If provided, saves the outputs to a CSV file.

    Returns:
    - pd.DataFrame containing the generated commentaries.
    """
    results = {}
    for path, data in descriptions.items():
        with torch.no_grad():
            commentator = Commentator()
            action_label = data["label"]
            action_description = data["description"]

            print(f"\n🎙 Generating commentary for {action_label}...")

            # Construct the LLM prompt
            # llm_prompt = f"""
            #                 You are a professional live sports commentator. 
            #                 Your job is to generate a natural, emotionally engaging **live soccer commentary** based on the given raw action description. Your commentary will be **converted into an audio track using a text-to-speech model**.

            #                 **Important Guidelines**:
            #                 - **IMAGINE YOU ARE LIVE ON AIR** describing this moment in real time.
            #                 - **Follow the EXACT CHRONOLOGY of the action with the TIMESTAMPS**, as if you were watching and reacting to it live.
            #                 - Your output must be **AS NATURAL, EMOTIONAL and HUMAN as POSSIBLE**, with the **RIGHT LEVEL OF EXCITEMENT**: 
            #                     - **BIG EVENTS (GOALS, PENALTIES, SHOTS ON TARGET, LAST-MINUTE ACTIONS)** should be **HIGHLY ENERGETIC** with shouting and excitement.
            #                     - **LOW-STAKES EVENTS (THROW-INS, FOULS, SUBSTITUTIONS)** should be **MORE DESCRIPTIVE AND NEUTRAL, WITH FEWER EXCLAMATION MARKS**.
            #                 - When there is a GOAL, a PENALTY or a SHOT, you can SHOUT but make it clear in the commentary so the text-to-speech model can transcribe it
            #                 - **Your commentary must fit exactly within 10 seconds**, so BE CONCISE YET DESCRIPTIVE.
            #                 - Use **SHORT AND ENERGETIC SENTENCES**, avoid unnecessary introductions.
            #                 - **Use TEXT FORMATTING to create expressive speech**:
            #                     - **SHOUT important words**: `"GOOOOAAALLL!!!"`, `"WHAT A STRIKE!"`, `"UNBELIEVABLE SAVE!!!"`
            #                     - **Elongate dramatic words**: `"GOOOOAAAAAAL!"`, `"NOOOOO! HE MISSED!!!"`
            #                     - **Use suspenseful pauses**: `"Messi... takes a step... SHOOTS—GOOOOOAAAALLL!!!"`
            #                     - **Use sound-like words** for realism: `"OH WOW! WHAT A STRIKE! THE CROWD ERUPTS!!!"`
            #                 - **Generate only one version** of the commentary—do not provide multiple variations or repeat the same event. **DON'T REPEAT THE TIMESTAMPS** you see in the description. **DO NOT include extra labels like 'Commentary:' or explanations about what you are doing.**
            #                 - USE THE MATCH_CONTEXT and the JERSEY-to-TEAM association to name the players (e.g. the Tottenham player). If a DESCRIBED COLOR DOESN'T MATCH with the MATCH_CONTEXT, don't use it. 
            #                 - Make sure you COVER THE WHOLE DESCRIPTION. You need to cover the full action, you can use the given timestamps to comment until the end of the action. IF SOME DESCRIPTION PARTS SEEM USELESS, DON'T USE THEM.

            #                 **Example Style (for a penalty goal)**:
            #                 - `"The Tottenham player steps up... he shoots—GOOOOOAAAAAALLLLL!!! The keeper dives the wrong way! WHAT A FINISH!!!"`
            #                 - `"Oh, what a pass! The striker controls it—LEFT FOOT SHOT—AND IT’S INNNNN!!!"`
            #                 - `"The ball is crossed in... header! It bounces—THE GOALKEEPER CAN'T REACH IT!!! INCREDIBLE!!!"`


            #                 Here is some context about the match you are commenting. USE THE JERSEY COLORS TO NAME THE TEAMS.
            #                 MATCH_CONTEXT: {match_context}

            #                 **Now, generate exactly ONE 10-second commentary (MAXIMUM 6 to 8 SENTENCES) for the following action description:**
            #                 '{action_description}'
            #                 """
            
            llm_prompt = f"""
            You are a professional live soccer commentator. 
            Your task is to generate a vivid and emotionally engaging **10-second spoken commentary** based on a detailed frame-by-frame action description. 

            Imagine you're broadcasting live: your words will be **converted into audio** using text-to-speech, so speak naturally, with excitement and clarity.

            ---

            **Guidelines**:
            - Follow the **CHRONOLOGY OF ACTIONS CLOSELY**, reflecting the sequence described (use the **timestamps as implicit structure**, but **don’t include them** in your output). DON'T REVEAL the OUTCOME of the action BEFORE IT HAPPENS.
            - Make sure you COVER THE DESCRIPTION CHRONOLOGICALLY. You can USE THE GIVEN TIMESTAMPS to comment until the end of the action. USE ONLY THE DESCRIPTIONS BETWEEN 0 AND 10 SECONDS.
            - Your commentary must be **live, energetic, and natural**. 
            - FIT EVERYTHING WITHIN 10 SECONDS — aim for **5 to 6 VERY SHORT AND CONCISE SENTENCES**.
            - **Your commentary must fit exactly within 10 seconds**, so USE CONCISE YET NATURAL SENTENCES.
            - DON'T PUT ANY STAR *
            - JUST GIVE THE OUTPUT, avoid unnecessary introduction. **DO NOT include extra labels like 'Commentary:' or explanations about what you are doing.**

            - Jersey color is linked to team identity. USE TEAM NAMES INSTEAD OF JERSEY COLOR from this MATCH CONTEXT:
            {match_context}

            **Tone**:
            - ONLY SAY GOOAALL IF THERE IS ACTUALLY A GOAL
            -**Use TEXT FORMATTING to create expressive speech** for SUSPENSEFUL ACTIONS:
                - **SHOUT important words**: `"GOOAALL!!!"`, `"WHAT A STRIKE!"`, `"UNBELIEVABLE SAVE!!!"`
                - **Elongate dramatic words**: `"GOOAAAL!"`, `"NOOOO! HE MISSED!!!"`
                - **Use suspenseful pauses**: `"Messi... takes a step... SHOOTS—GOOAAALLL!!!"`
                - **Use sound-like words** for realism: `"OH WOW! WHAT A STRIKE! THE CROWD ERUPTS!!!"`
            - However don't try to name the players, DON'T put things like [Player Name]

            ---

            **IMPORTANT**:
            - Cover the full action step-by-step.
            - **Don’t summarize**. Capture the evolving intensity.
            - **Only output the final commentary**, no metadata, no labels.

            ---

            Here is the action description to turn into a 10-second commentary (MAXIMUM 5 TO 6 SENTENCES - MAXIMUM 200 characters):
            '{action_description}'
            """

            # llm_prompt = f"""
            # You are a professional live soccer commentator. 
            # Your task is to generate a vivid and emotionally engaging **10-second spoken commentary** based on a detailed frame-by-frame action description. 

            # **Guidelines**:
            # - Follow the **CHRONOLOGY OF ACTIONS CLOSELY**, reflecting the sequence described (use the **timestamps as implicit structure**, but - Your commentary must be **live, energetic, and natural**. 

            # - Jersey color is linked to team identity. USE TEAM NAMES INSTEAD OF JERSEY COLOR from this MATCH CONTEXT:
            # {match_context}

            # HERE, action_label = {action_label}

            # Here is the action description to turn into a 30 words commentary MAXIMUM:):
            # '{action_description}'
            # """
            

            commentary = commentator.rewrite_text(llm_prompt)
            print(f"len de commentary = {len(commentary.split(' '))}")
            results[path] ={
                    "Action": action_label,
                "Raw Description": action_description,
                "Generated Commentary": commentary
            }
        del commentator, commentary, llm_prompt
        torch.cuda.empty_cache()
        gc.collect()

    return results

# Example Usage:
if __name__ == "__main__":
    # Example match context
    match_context = "It's a Premier League match between Tottenham and Chelsea. Tottenham plays with white jerseys, and Chelsea plays with dark blue jerseys."

    # Generate commentary and save to CSV
    #results = generate_commentary(DESCRIPTIONS_OUTPUT, match_context)

    # Print the generated commentaries
    #print("\n📋 Generated Commentaries:")
    #print(results)


