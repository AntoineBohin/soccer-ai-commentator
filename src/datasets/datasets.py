import abc
import random
from typing import Callable, Type, Optional

import numpy as np

import torch

from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.data_processing.frame_fetchers import AbstractFrameFetcher, NvDecFrameFetcher
from src.utils import set_random_seed
from src.datasets.target import VideoTarget


class ActionDataset(metaclass=abc.ABCMeta):
    """
    Base class for loading video datasets for action spotting.
    Implements logic for loading frames, labels, and preprocessing.
    """
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
            indexes_generator: StackIndexesGenerator,
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Initializes the dataset class.
        Args:
            videos_data (list[dict]): Metadata for each video.
            classes (list[str]): List of possible action labels.
            indexes_generator (StackIndexesGenerator): Generates frame indexes for each clip.
            target_process_fn (Callable): Function that processes target labels.
            frames_process_fn (Callable): Function that processes video frames.
        """
        self.indexes_generator = indexes_generator
        self.frames_process_fn = frames_process_fn
        self.target_process_fn = target_process_fn

        self.videos_data = videos_data
        self.num_videos = len(self.videos_data)
        self.num_videos_actions = [len(v["frame_index2action"]) for v in self.videos_data]
        self.num_actions = sum(self.num_videos_actions)
        self.videos_target = [
            VideoTarget(data, classes) for data in self.videos_data
        ]

    def __len__(self) -> int:
        """
        Returns the total number of labeled actions in the dataset.
        """
        return self.num_actions

    @abc.abstractmethod
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        pass

    def get_targets(self, video_index: int, frame_indexes: list[int]):
        """
        Retrieves action labels for a given set of frame indexes.
        Args:
            video_index (int): Index of the video.
            frame_indexes (list[int]): List of frame indexes.
        Returns:
            np.ndarray: One-hot encoded labels for each frame in the clip.
        """
        # Generate the full list of frame indexes for the given range
        target_indexes = list(range(min(frame_indexes), max(frame_indexes) + 1))
        # Retrieve action labels from VideoTarget
        targets = self.videos_target[video_index].targets(target_indexes)
        return targets

    def get_frames_targets(
            self,
            video_index: int,
            frame_indexes: list[int],
            frame_fetcher: AbstractFrameFetcher
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Fetches frames and their corresponding action labels.
        Args:
            video_index (int): Index of the video.
            frame_indexes (list[int]): List of frame indexes to fetch.
            frame_fetcher (AbstractFrameFetcher): Object for loading frames from video.
        Returns:
            tuple: (frames tensor, targets numpy array)
        """
        frames = frame_fetcher.fetch_frames(frame_indexes) # Load frames from video
        targets = self.get_targets(video_index, frame_indexes) # Get labels for frames
        return frames, targets

    def get_frame_fetcher(self,
                          video_index: int,
                          frame_fetcher_class: Type[AbstractFrameFetcher],
                          gpu_id: int = 0):
        """
        Creates a frame fetcher object to load frames from a video.
        Args:
            video_index (int): Index of the video.
            frame_fetcher_class (Type[AbstractFrameFetcher]): Class to use for frame fetching.
            gpu_id (int): GPU ID for processing (default is 0).
        Returns:
            AbstractFrameFetcher: Initialized frame fetcher.
        """
        video_data = self.videos_data[video_index]
        frame_fetcher = frame_fetcher_class(
            video_data["video_path"],
            gpu_id=gpu_id
        )
        frame_fetcher.num_frames = video_data["frame_count"] # Store total frame count
        return frame_fetcher

    def process_frames_targets(self, frames: torch.Tensor, targets: np.ndarray):
        """
        Applies preprocessing functions to frames and labels.
        Args:
            frames (torch.Tensor): Raw video frames.
            targets (np.ndarray): Raw action labels.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed inputs and labels.
        """
        input_tensor = self.frames_process_fn(frames)
        target_tensor = self.target_process_fn(targets)
        return input_tensor, target_tensor

    def get(self,
            index: int,
            frame_fetcher_class: Type[AbstractFrameFetcher] = NvDecFrameFetcher,
            gpu_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a processed video clip and its corresponding labels.
        Args:
            index (int): Index of the action.
            frame_fetcher_class (Type[AbstractFrameFetcher]): Class for frame fetching.
            gpu_id (int): GPU ID for processing.
        Returns:
            tuple: (Processed frames tensor, processed targets tensor)
        """
        video_index, frame_indexes = self.get_video_frame_indexes(index)
        frame_fetcher = self.get_frame_fetcher(video_index, frame_fetcher_class, gpu_id)
        frames, targets = self.get_frames_targets(video_index, frame_indexes, frame_fetcher)
        return self.process_frames_targets(frames, targets)


class TrainActionDataset(ActionDataset):
    """
    Dataset class for training.
    Handles random sampling of frames for training.
    """
    def __init__(
            self,
            videos_data: list[dict],
            classes: list[str],
            indexes_generator: StackIndexesGenerator,
            epoch_size: int,
            videos_sampling_weights: list[np.ndarray], # Probability distribution for selecting frames
            target_process_fn: Callable[[np.ndarray], torch.Tensor],
            frames_process_fn: Callable[[torch.Tensor], torch.Tensor],
            frame_index_shaker: Optional[FrameIndexShaker] = None, # Adds small random shifts to frames
    ):
        super().__init__(
            videos_data=videos_data,
            classes=classes,
            indexes_generator=indexes_generator,
            target_process_fn=target_process_fn,
            frames_process_fn=frames_process_fn,
        )
        self.epoch_size = epoch_size
        self.frame_index_shaker = frame_index_shaker
        # Store frame selection probabilities for each video
        self.videos_sampling_weights = videos_sampling_weights
        self.videos_frame_indexes = [np.arange(v["frame_count"]) for v in videos_data]

    def __len__(self) -> int:
        return self.epoch_size

    def get_video_frame_indexes(self, index) -> tuple[int, list[int]]:
        """
        Randomly selects a video and frame for training.
        Args:
            index (int): Training sample index.
        Returns:
            tuple[int, list[int]]: (Video index, list of frame indexes)
        """
        set_random_seed(index)
        video_index = random.randrange(0, self.num_videos) # Pick a random video
        frame_index = np.random.choice(self.videos_frame_indexes[video_index], p=self.videos_sampling_weights[video_index]) # Sample a frame
        save_zone = 1
        if self.frame_index_shaker is not None:
            save_zone += max(abs(sh) for sh in self.frame_index_shaker.shifts)
        frame_index = self.indexes_generator.clip_index(
            frame_index, self.videos_data[video_index]["frame_count"], save_zone
        ) # Clip to valid range
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)  # Generate frame sequence
        if self.frame_index_shaker is not None:
            frame_indexes = self.frame_index_shaker(frame_indexes)
        return video_index, frame_indexes


class ValActionDataset(ActionDataset):
    """
    Dataset class for validation.
    Uses fixed frame indexes for reproducibility.
    """
    def get_video_frame_indexes(self, index: int) -> tuple[int, list[int]]:
        assert 0 <= index < self.__len__()
        action_index = index
        video_index = 0
        for video_index, num_video_actions in enumerate(self.num_videos_actions):
            if action_index >= num_video_actions:
                action_index -= num_video_actions
            else:
                break
        video_target = self.videos_target[video_index]
        video_data = self.videos_data[video_index]
        frame_index = video_target.get_frame_index_by_action_index(action_index)
        frame_index = self.indexes_generator.clip_index(frame_index, video_data["frame_count"], 1)
        frame_indexes = self.indexes_generator.make_stack_indexes(frame_index)
        return video_index, frame_indexes


