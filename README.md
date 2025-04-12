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
We used **SoccerNet v2**, a public benchmark dataset for soccer video understanding.

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

<video src="https://github.com/user-attachments/assets/39a5404e-2ff0-400b-b9f7-7891efc3f97d" controls width="40%">


### 3. Action Description (VLM)

### 4. Textual Commentary (LLM)

### 5. Audio Commentary (TTS)

## Quick Setup and Start

## Contributors
- Antoine Bohin (CentraleSupélec) antoine.bohin@student-cs.fr
- Ilyess Doragh (CentraleSupélec) ilyess.doragh@student-cs.fr
- Mathieu Dujardin (CentraleSupélec) mathieu.dujardin@student-cs.fr
- Logan Renaud (CentraleSupélec / MVA) logan.renaud@student-cs.fr
- 
