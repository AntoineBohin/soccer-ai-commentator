from typing import Any
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms as transforms

from src.data_processing.frame_fetchers.abstract import AbstractFrameFetcher

class NvDecFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self.video_path = str(video_path)

        # Use `video_reader` from torchvision for GPU-accelerated decoding
        self.video_reader = tvio.VideoReader(self.video_path, "video")
        self.video_reader.set_current_stream("video")
        
        # Get video properties
        self.width = 1280
        self.height = 720
        self.fps = 25
        self.duration = 771#2765
        
        self.num_frames = int(self.duration * self.fps)
        self._current_index = 0  # Frame index starts from 0
        
        # Define grayscale transform
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)
    
    def estimate_duration(self):
        """Estimate duration based on available frames or default value."""
        try:
            frame_count = sum(1 for _ in self.video_reader)
            estimated_duration = frame_count / self.fps
            print(f"Estimated duration: {estimated_duration:.2f} seconds")
            return estimated_duration
        except Exception as e:
            print(f"Failed to estimate duration: {e}")
            return 2765  # Hardcoded duration from ffmpeg output

    def _next_decode(self) -> Any:
        """Fetches the next frame"""
        frame = next(self.video_reader)
        return frame["data"].to(f"cuda:{self.gpu_id}")  # Ensure it's on GPU

    def _seek_and_decode(self, index: int) -> Any:
        """Seeks a specific frame index and decodes it"""
        self.video_reader.seek(index / 25)
        frame = next(self.video_reader)
        return frame["data"].to(f"cuda:{self.gpu_id}")  # Ensure GPU storage

    def _convert(self, frame: Any) -> torch.Tensor:
        """Converts frame to grayscale"""
        frame = self.to_grayscale(frame)  # Apply grayscale conversion
        frame=frame.squeeze(0)
        if frame.dim() == 2:  
            frame = frame.unsqueeze(0)  # Add channel dimension if missing
        return frame

"""
class NvDecFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path=video_path, gpu_id=gpu_id)
        self.video_path = str(video_path)

        # Use `video_reader` from torchvision for GPU-accelerated decoding
        self.video_reader = tvio.VideoReader(self.video_path, "video")
        self.video_reader.set_current_stream("video")
        
        # Get video properties
        self.num_frames = int(self.video_reader.get_metadata()["video"]["duration"] * 
                              self.video_reader.get_metadata()["video"]["fps"][0])
        self.width = int(self.video_reader.get_metadata()["video"]["width"])
        self.height = int(self.video_reader.get_metadata()["video"]["height"])

        self._current_index = 0  # Frame index starts from 0
        
        # Define grayscale transform
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)

    def _next_decode(self) -> Any:
        frame = next(self.video_reader)
        return frame["data"].to(f"cuda:{self.gpu_id}")  # Ensure it's on GPU

    def _seek_and_decode(self, index: int) -> Any:
        self.video_reader.seek(index / self.video_reader.get_metadata()["video"]["fps"][0])
        frame = next(self.video_reader)
        return frame["data"].to(f"cuda:{self.gpu_id}")  # Ensure GPU storage

    def _convert(self, frame: Any) -> torch.Tensor:
        frame = self.to_grayscale(frame)  # Apply grayscale conversion
        frame = frame.squeeze(0)  # Remove unnecessary dimension
        frame = frame.permute(1, 2, 0)  # Rearrange dimensions if needed
        return frame
"""