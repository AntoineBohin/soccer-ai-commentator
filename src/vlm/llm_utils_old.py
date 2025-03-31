from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
import re


class Commentator:
    def __init__(self):
        """
        Initialize the Qwen2.5-7B-Instruct model for text generation.
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

        print("Loading model (16-bit mode activated)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Load in 16-bit precision
            device_map="auto"  # Auto-distribute across GPU/CPU
        )

        print("Configuring text generation pipeline...")
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def rewrite_text(self, prompt, max_length=500):
        """
        Reformulates a detailed VLM description into a concise and professional sports commentary.
        """
        print("Generating commentary...")

        result = self.generator(
            prompt,
            temperature=0.7,  # Balances creativity and realism
            do_sample=True,
            max_new_tokens=max_length,  # Allows for longer but controlled outputs
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure it stops at EOS
            eos_token_id=self.tokenizer.eos_token_id,  # Hard stop when EOS is reached
            return_full_text=False
        )

        # Extract the generated text
        answer = result[0]["generated_text"].strip()

        # Remove the input prompt from the output if included
        #answer_without_prompt = answer[len(prompt):].strip()
        answer_without_prompt = answer

        return answer_without_prompt


def generate_commentary(descriptions, match_context, num_variations=3, save_path=None):
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
    commentator = Commentator()
    results = []

    for game_time, data in descriptions.items():
        action_label = data["label"]
        action_description = data["description"]

        print(f"\n🎙 Generating commentary for {action_label} at {game_time}...")

        # Construct the LLM prompt
        llm_prompt = (
            f"You are a professional live sports commentator. \n"
            f"Your job is to generate a natural, emotionally engaging **live soccer commentary** based on the given raw action description.\n\n"
            f"**Important Guidelines**:\n"
            f"- **IMAGINE YOU ARE LIVE ON AIR** describing this moment in real time.\n"
            f"- Your commentary will be **converted into an audio track using a text-to-speech model**.\n"
            f"- Your output must be **AS NATURAL, EMOTIONAL and HUMAN as POSSIBLE**, with the **RIGHT LEVEL OF EXCITEMENT**.\n"
            f"- **Your commentary must fit exactly within 10 seconds**, so BE CONCISE YET DESCRIPTIVE.\n"
            f"- **Follow the exact chronology of the action**, as if you were watching and reacting to it live.\n"
            f"- Use **VERY SHORT AND ENERGETIC SENTENCES**, avoid unnecessary introductions.\n"
            f"- **Generate only one version** of the commentary—do not provide multiple variations or repeat the same event.\n"
            f"- **DO NOT include extra labels like 'Commentary:' or explanations about what you are doing.**\n"
            f"- **DON'T REPEAT THE TIMESTAMPS** you see in the description.\n"
            f"- USE THE MATCH_CONTEXT and the JERSEY-to-TEAM association to name the players (e.g. the Tottenham player). If a DESCRIBED COLOR DON'T MATCH with the MATCH_CONTEXT, don't use it. \n"
            f"- Make sure you COVER THE WHOLE DESCRIPTION. You need to cover the full action, you can use the given timestamps to you comment until the end of the action. IF SOME DECRIPTION PARTS SEEMS TO BE USELESS, DON'T USE THEM\n\n"
            f"**Example Style (for a penalty goal)**:\n"
            f"- 'The Tottenham player steps up... he shoots—GOOOAL! The keeper dives the wrong way! What a finish!'\n"
            f"- 'Oh, what a pass! The striker controls it—LEFT FOOT SHOT—AND IT’S IN!'\n"
            f"- 'The ball is crossed in... header! It bounces—THE GOALKEEPER CAN'T REACH IT! Incredible!'\n\n"
            f"Here is some context about the match you are commenting,:\n"
            f"MATCH_CONTEXT: {match_context}"
            f"**Now, generate exactly ONE 10-second commentary (MAXIMUM 6 TO 8 SENTENCES) for the following action description:**\n"
            f"'{action_description}'"
        )

        for i in range(num_variations):
            commentary = commentator.rewrite_text(llm_prompt)

            # Extract only the first 1-2 sentences
            #sentences = re.findall(r'[^.!?]+[.!?]', commentary)
            #best_commentary = ' '.join(sentences[:2])

            print(f"🗣️ Commentary ({i+1}): {commentary}")

            results.append({
                "Time": game_time,
                "Action": action_label,
                "Raw Description": action_description,
                "Generated Commentary": commentary
            })

    return results

# Example Usage:
if __name__ == "__main__":
    # Example match context
    match_context = "It's a Premier League match between Tottenham and Chelsea. Tottenham plays with white jerseys, and Chelsea plays with dark blue jerseys."

    # Example descriptions dictionary from the VLM output
    descriptions = {
        '1 - 01:34': {'label': 'Throw-in', 'description': '**0-2 seconds:** The throw-in begins with the player in the defending team (wearing white jerseys) preparing to throw the ball back into play. The player is positioned near the corner flag, holding the ball securely with both hands. The ball is thrown into the field of play, and the camera follows its trajectory.\n\n**2-4 seconds:** The ball lands near the center of the field, and the attacking team (wearing blue jerseys) quickly gathers around it. The player in possession of the ball (blue jersey) starts to control it, using their feet to direct the ball away from the defending team. The defending team players are moving towards the ball, attempting to intercept it.\n\n**4-6 seconds:** The player in the blue jersey passes the ball to a teammate who is positioned further up the field. The pass is made with a short, controlled kick, and the ball travels across the field. The defending team players continue to move forward, trying to press and regain possession.\n\n**6-8 seconds:** The ball reaches the receiving player in the blue jersey, who is now in a more advanced position. This player takes a few steps forward and then kicks the ball back towards the center of the field. The defending team players are still moving towards the ball, but they are slightly behind the attacking team.\n\n**8-10 seconds:** The ball is kicked by the blue-jerseyed player and travels across the field again. It lands near the center of the field, and the defending team players are now positioned closer to it. The ball is then intercepted by one of the defending players, who gains possession and starts to move the ball back towards their own goal.'},
        '1 - 02:07': {'label': 'Goal', 'description': '**0-2 seconds:** The ball is in play near the center of the field. A player in a red jersey (Team A) receives the ball and begins to dribble forward. Another player in a blue jersey (Team B) is positioned nearby, observing the movement.\n\n**2-4 seconds:** As Team A advances the ball, a defender in a blue jersey (Team B) starts to close in, attempting to press the attacker. The ball is passed quickly between two players in red jerseys, maintaining possession and advancing the attack.\n\n**4-6 seconds:** The ball reaches a forward in a red jersey who is positioned near the edge of the penalty area. This player takes a few steps back and then accelerates towards the goal, preparing to take a shot.\n\n**6-8 seconds:** The forward in red approaches the penalty spot and takes a powerful shot at the goal. The goalkeeper in a green jersey (Team B) dives to make a save but fails to stop the ball.\n\n**8-10 seconds:** The ball enters the net, and the referee raises his arm to signal a goal. Players from both teams react; Team A celebrates the goal with excitement, while Team B shows disappointment. The ball remains in the net, and the match continues with the next phase of play.'}
    }

    # Generate commentary and save to CSV
    results = generate_commentary(descriptions, match_context, num_variations=3, save_path="generated_commentary.csv")

    # Print the generated commentaries
    #print("\n📋 Generated Commentaries:")
    #print(results)
