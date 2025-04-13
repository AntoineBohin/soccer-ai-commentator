# AI-Generation of Soccer Highlights and Audio Commentaries

This project presents an end-to-end AI pipeline for automatically generating natural audio commentary from raw soccer match videos. Combining the latest advances in Computer Vision, Vision-Language Modeling (VLM), Large Language Models (LLM), and Text-to-Speech (TTS), the pipeline detects key actions, describes them chronologically, reformulates them into sports commentary, and synthesizes spoken audio.

## Context

Generating automated, high-quality soccer commentary from full-match videos is a highly relevant challenge with real-world applications, including YouTube highlights, automated post-match reporting, and textual recaps for sports media like L’Équipe. However, current multimodal AI models face limitations that prevent end-to-end solutions: video inputs must be drastically reduced due to token and memory constraints, and existing Vision-Language Models (VLMs) lack fine temporal understanding, narrative coherence, and emotional tone.

To address this, we designed a modular and scalable pipeline combining vision, language, and audio generation — tailored to work under hardware constraints, yet easily upgradable as models evolve. Our approach breaks down the problem into specialized steps, allowing for flexible customization (e.g., prompts, FPS, resolution), rapid model replacement, and potential adaptation to other sports like basketball or e-sports. This project demonstrates the feasibility of such a system, offering a valuable foundation for future research and product development in sports analytics, media production, and AI-generated storytelling.

<video src="https://github.com/user-attachments/assets/9ef1e9a8-7c0a-42a7-8ea9-f4ccabebc151" controls width="100%"></video> <video src="https://github.com/user-attachments/assets/5643d22c-89cb-4ca2-a31a-4694f95e1d92" controls width="100%"></video>
You can watch the full demo of the solution by downloading Final_Demo.mp4

## Project Overview
The pipeline consists of 5 main components, described below. \\
![Project Pipeline](./img/pipeline.png)

### Soccer Data
We used **SoccerNet v2**, a public benchmark dataset for soccer video understanding. This dataset served as the foundation for training and evaluating our models — particularly the Action Spotting and Tracking modules — while ensuring comparability with state-of-the-art approaches.

Key features:
- 550+ full matches (~764 hours) from 6 major European leagues
- Over 110,000 annotated events across 17 action classes (e.g., goal, foul, card, substitution, corner…)
- Videos available in 720p and 224p, with frame-level annotations and timestamps
- Fully accessible via the official Python API

Why [SoccerNet](https://www.soccer-net.org)?
- Recognized as the benchmark dataset in Computer Vision applied to Soccer
- Hosts annual international challenges (CVPR, ECCV…) on tasks like Action Spotting, Replay Grounding, Multi-View Detection, etc.
- Strong academic relevance: +5 Best Paper Awards (CVPR, ECCV), +60 published methods for Action Spotting
- Practical applications: Performance analysis, TV broadcasting, scouting, VAR augmentation

### 1. Action Spotting
We selected an open-source model from the 2023 SoccerNet Action Spotting Challenge, originally developed by [Ruslan Baikulov (original github)](https://github.com/lRomul/ball-action-spotting). This model (that has become the standard baseline for SoccerNet 2024 Action Spotting challenge) belongs to a class of hybrid and lightweight Action Spotting architectures, offering a solid compromise between performance, robustness, and efficiency. Its design makes it particularly well-suited for deployment on modest GPUs, a crucial criterion for scalable and accessible solutions.

The model works in three stages:
- Visual Encoding: Consecutive grayscale frames are stacked and passed through a 2D CNN backbone (EfficientNetV2-B0) to extract spatial features.
- Temporal Modeling: A lightweight 3D CNN merges motion cues across time to capture action dynamics.
- Classification: A linear layer classifies each timestep into one of the SoccerNet action classes.

We reused and refactored the original implementation to adapt it to our project structure:
- Model architecture and training logic: available under `src/`.
- Training and inference scripts: located in `scripts/`, with clear CLI options for configuration.
- Model weights available in `data/action/`
- The model is fully compatible with SoccerNet v2 and can be retrained or used for direct inference.

### 2. Player & Ball Tracking
To improve visual understanding and context for downstream models, we implemented a lightweight two-step tracking system, adapted from the [Roboflow Sports GitHub repository](https://github.com/roboflow/sports/tree/main).
1. Detection: We use the YOLOv8 object detection model to detect players, referees, and the ball in each frame with high speed and precision.
2. Tracking: Detections are linked across frames using ByteTrack, a high-performance multi-object tracking algorithm based on IoU matching and a two-stage association of low and high confidence detections.

This combination allows us to:
- Persistently track individual players and the ball across time,
- Assign unique IDs and team colors,
- Provide structured spatial and temporal context to the Vision-Language Model (VLM).
Code for tracking can be found in `src/player_tracking/` (modular pipeline adapted from Roboflow)

This approach enables the VLM to focus attention on relevant actors and motions, improving both action classification and description generation.

<video src="https://github.com/user-attachments/assets/39a5404e-2ff0-400b-b9f7-7891efc3f97d" controls width="40%"></video>

### 3. Action Description (VLM)
To describe soccer actions from video clips, we tested and deployed several open-source Vision-Language Models (VLMs) locally — without relying on external APIs.
Given the GPU memory constraints (max 24 GB VRAM), we designed a lightweight inference strategy:
- Compact open-source models only
- Use of FlashAttention and float16 inference to optimize memory and speed
- Extensive benchmark of VLMs on football action clips (Phi-3.5 Vision, SmolVLM 2.2B, Apollo-7B, PaliGemma-3B, InternVL2.5, IBM Granite)
	→ Winner: QWEN2.5-VL-7B for its overall performance and low hallucination rate

Due to strict token and memory limits, we optimized video input as follows:
- 10s clips, downsampled to 4 fps (preserves motion)
- Resolution reduced to match GPU memory limits (c.220p with ours), while keeping player & ball tracking overlays for context
- Best results came from clips with varied viewpoints and player close-ups

Well-crafted prompts were critical to VLM performance. Ours included:
- Action label (from Action Spotting)
- Frame count and timestamps
- Request to describe the clip by time chunks (e.g., 1s blocks)

This led to temporally aligned outputs suitable for rephrasing and TTS. Complex instructions (e.g., asking for audio-ready commentary directly) reduced reliability.
Code for VLM can be found in `src/vlm/vlm_utils.py`

### 4. Textual Commentary (LLM)
After generating raw descriptions with the VLM, we use a Large Language Model (LLM) to rephrase them into fluent, expressive sport commentaries. Since the LLM does not process videos directly, we can afford to use larger models optimized for text generation. Among various tested models (GPT-2, Mistral-7B, Qwen2.5-7B, Gemma 3-4B), we selected Gemma 3-4B Instruct for its strong balance between creativity and reliability (2% hallucination rate).
The LLM receives a structured prompt that helps:
- Add emotion and storytelling to the descriptions,
- Reduce hallucinations from the VLM,
- Maintain a short, coherent summary (~30 words).

### 5. Audio Commentary (TTS)
The final step of our pipeline transforms textual descriptions into natural, expressive audio commentary, simulating a real sports broadcast experience.
We use Zonos, a Text-to-Speech model from the Coqui-TTS framework, selected for its open-source nature, fast inference, and advanced control over style and emotion. It allows us to specify voice tone parameters dynamically, making it especially suited for sports applications.
Key features of our setup:
- Voice cloning: We use real audio samples of Peter Drury, a renowned sports commentator, to synthesize voice with high fidelity and recognizable tone.
- Emotion embeddings: We encode emotions like happiness, excitement, or neutrality as a vector and pass it to the TTS model to modulate the speaking style.
- Context-aware style: Two different voice inputs and sets of emotion parameters are used based on the action type: High-emotion voice for impactful actions (e.g. goals, penalties) vs. Neutral tone for routine actions (e.g. throw-ins, substitutions).
- Low inference time: The full generation process is typically under 10 seconds per clip on a standard GPU.
You can test Zonos and tune voice parameters on their playground: [https://github.com/roboflow/sports/tree/main](https://playground.zyphra.com/audio)

## Quick Setup and Start


## Contributors
- Antoine Bohin (CentraleSupélec) antoine.bohin@student-cs.fr
- Ilyess Doragh (CentraleSupélec) ilyess.doragh@student-cs.fr
- Mathieu Dujardin (CentraleSupélec) mathieu.dujardin@student-cs.fr
- Logan Renaud (CentraleSupélec / MVA) logan.renaud@student-cs.fr
- Project proposed by Sia Partners (Axel Darmouni and Guillaume Orset-Prelet)
