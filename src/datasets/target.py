"""Processing action labels for video-based classification. It converts frame-based event annotations into structured target tensors suitable for training a neural network"""

import abc
from collections import defaultdict

import torch
import numpy as np


class VideoTarget:
    """
    This class manages action labels for a video dataset.
    It stores frame-to-action mappings and provides functions to retrieve labels for training.
    """
    def __init__(self, video_data: dict, classes: list[str]):
        """
        Initializes the VideoTarget class.
        Args:
            video_data (dict): Contains video metadata and action annotations.
            classes (list[str]): List of possible action classes.
        """
        # List of all action classes
        self.classes = classes
        self.num_classes = len(classes)

        # Mapping from class names to indices (e.g., {"goal": 0, "pass": 1})
        self.class2target = {cls: trg for trg, cls in enumerate(classes)}
         # Dictionary that maps each class to a frame index, defaulting to 0.0 (no event)
        self.frame_index2class_target: dict[str, defaultdict] = {
            cls: defaultdict(float) for cls in classes
        }
        # Dictionary that maps action indices to corresponding frame indices
        self.action_index2frame_index: dict[int, int] = dict()
        # Sorts the actions by frame index to process them in order
        actions_sorted_by_frame_index = sorted(
            video_data["frame_index2action"].items(), key=lambda x: x[0]
        )
        # Populate mappings: Assigns each action to the correct frame index
        for action_index, (frame_index, action) in enumerate(actions_sorted_by_frame_index):
            self.action_index2frame_index[action_index] = frame_index
            if action in classes:
                # Mark that an action happens at this frame
                self.frame_index2class_target[action][frame_index] = 1.0

    def target(self, frame_index: int) -> np.ndarray:
        """
        Returns a one-hot encoded vector for the given frame.
        Args:
            frame_index (int): Frame index to retrieve labels.
        Returns:
            np.ndarray: One-hot encoded array of shape (num_classes,).
        """
        # Initialize a zero vector for all classes
        target = np.zeros(self.num_classes, dtype=np.float32)
        # Set 1.0 if an event occurs at the frame
        for cls in self.classes:
            target[self.class2target[cls]] = self.frame_index2class_target[cls][frame_index]
        return target

    def targets(self, frame_indexes: list[int]) -> np.ndarray:
        """
        Returns one-hot encoded labels for a sequence of frames.
        Args:
            frame_indexes (list[int]): List of frame indices.
        Returns:
            np.ndarray: Stacked target vectors.
        """
        # Retrieve target labels for each frame and stack them
        targets = [self.target(idx) for idx in frame_indexes]
        return np.stack(targets, axis=0)

    def get_frame_index_by_action_index(self, action_index: int) -> int:
        """
        Returns the frame index corresponding to an action index.
        """
        return self.action_index2frame_index[action_index]

    def num_actions(self) -> int:
        """
        Returns the total number of actions in the video.
        """
        return len(self.action_index2frame_index)


# =======================
# TARGET PROCESSING UTILS
# =======================

def center_crop_targets(targets: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Crops the center portion of the target sequence.
    Args:
        targets (np.ndarray): Target array of shape (num_frames, num_classes).
        crop_size (int): Desired length of cropped sequence.
    Returns:
        np.ndarray: Cropped target sequence.
    """
    num_crop_targets = targets.shape[0] - crop_size
    left = num_crop_targets // 2
    right = num_crop_targets - left
    return targets[left:-right]



class TargetsToTensorProcessor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        pass


class MaxWindowTargetsProcessor(TargetsToTensorProcessor):
    """
    Converts target labels into PyTorch tensors and applies max-pooling.
    This helps to keep the most confident event activations.
    """
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, targets: np.ndarray) -> torch.Tensor:
        targets = targets.astype(np.float32, copy=False)
        targets = center_crop_targets(targets, self.window_size)
        target = np.amax(targets, axis=0)
        target_tensor = torch.from_numpy(target)
        return target_tensor