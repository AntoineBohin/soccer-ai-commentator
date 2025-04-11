import os
#os.environ["HF_TOKEN"]

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re
import pandas as pd

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

#vlm_result="**0-2 seconds:** The corner kick is taken by a player wearing a white jersey. The ball is kicked into the penalty area with significant force. The goalkeeper, dressed in a dark blue jersey, is positioned near the goalpost, ready to intercept the ball.\n\n**2-4 seconds:** The ball travels through the air and lands near the edge of the penalty box. A player in a red jersey is seen running towards the ball, attempting to head it. Another player in a white jersey is positioned nearby, preparing to react to the header.\n\n**4-6 seconds:** The player in the red jersey successfully heads the ball, sending it flying towards the goal. The goalkeeper in the dark blue jersey dives to make a save but misses the ball, which continues towards the goal.\n\n**6-8 seconds:** The ball rolls towards the goal line, and a player in a white jersey is seen running towards it. Another player in a red jersey is close behind, trying to intercept the ball. The ball is eventually kicked out of play by a player in a white jersey who is outside the penalty area.\n\n**8-10 seconds:** The ball is cleared away from the goal area by a player in a white jersey. The players in both teams gather around the corner flag, preparing for the next phase of play. The referee signals the end of the corner kick, and the game resumes with a throw-in from the opposite side of the field."
vlm_result='*0-2 seconds:* The ball is initially played by a Tottenham player in a white jersey, who is positioned near the center of the field. He receives the ball with his right foot and begins to dribble forward, looking to pass or shoot. Chelsea players in dark blue jerseys are positioned around him, attempting to press and intercept the ball.\n\n*2-4 seconds:* As the Tottenham player continues to dribble, he is closely marked by a Chelsea defender. The defender tries to challenge for the ball but fails to gain possession. The Tottenham player maintains control and continues advancing towards the goal area.\n\n*4-6 seconds:* The Tottenham player approaches the penalty box and decides to take a shot at goal. He takes a few steps back and strikes the ball with his right foot, sending it towards the goal.\n\n*6-8 seconds:* The ball travels through the air and heads towards the goal. The Chelsea goalkeeper, dressed in a dark blue jersey, dives to make a save but misses the ball completely. The ball goes into the net, resulting in a goal for Tottenham.\n\n*8-10 seconds:* The goal is confirmed by the referee, and the crowd erupts in celebration. Tottenham players run towards the goal to celebrate their score, while Chelsea players look dejected. The match continues with both teams now aware that Tottenham has taken the lead.'

llm_prompt = (
            f"You are a professional live sports commentator. \n"
            f"Your job is to generate a natural, emotionally engaging *live soccer commentary* based on the given raw action description.\n\n"
            f"*Important Guidelines*:\n"
            f"- *Imagine you are live on air* describing this moment in real time.\n"
            f"- Your commentary will be *converted into an audio track using a text-to-speech model*.\n"
            f"- Your output must be *as natural and human as possible, with the **right level of excitement*.\n"
            f"- *Your commentary must fit exactly within 10 seconds*, so be concise yet descriptive.\n"
            f"- *Follow the exact chronology of the action*, as if you were watching and reacting to it live.\n"
            f"- Use *short, energetic sentences*, avoid unnecessary introductions.\n"
            f"- *DON'T REPEAT THE TIMESTAMPS* you see in the description.\n"
            f"- USE THE MATCH_CONTEXT and the JERSEY-to-TEAM association to you can especially use the jersey colors of the description to name the teams\n"
            f"- MAKE SURE YOU USE THE WHOLE DESCRIPTION. You need to cover the full action, you can use the given timestamps to make sure you comment each part of the action.\n\n"
            f"*Example Style (for a penalty goal)*:\n"
            f"- 'Messi steps up... he shoots—GOOOAL! The keeper dives the wrong way! What a finish!'\n"
            f"- 'Oh, what a pass! The striker controls it—LEFT FOOT SHOT—AND IT’S IN!'\n"
            f"- 'The ball is crossed in... header! It bounces—THE GOALKEEPER CAN'T REACH IT! Incredible!'\n\n"
            f"Here is some context about the match you are commenting,:\n"
            f"MATCH_CONTEXT: It's a Premier League match between Tottenham and Chelsea. Tottenham plays with white jerseys, and Chelsea plays with dark blue jerseys."
            f"*Now, transform the following action description into a 10-second real-time commentary:*\n"
            f"{vlm_result}"
        )


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful sports commentator."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{llm_prompt}"}
        ]
    }
]

answers = []
nb_iterations = 3

for i in range(nb_iterations):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=True)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(f"Iteration {i+1}: {decoded}\n")
    answers.append(decoded)

# Sauvegarde des réponses dans un CSV
df = pd.DataFrame(answers, columns=["Answers"])
# df.to_csv("action_4B.csv", index=False)
