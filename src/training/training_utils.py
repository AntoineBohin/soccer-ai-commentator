import os
import shutil
import torch
import numpy as np

class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    """

    def __init__(self, name: str, fmt=':f'):
        """
        Initializes the AverageMeter.

        Args:
            name (str): Name of the metric.
            fmt (str): Format for printing.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Resets the metric values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the metric.

        Args:
            val (float): New value.
            n (int): Number of samples the value corresponds to.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        """Returns a formatted string representation."""
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


class ProgressMeter:
    """
    Tracks training progress and logs output.
    """

    def __init__(self, prefix: str, num_batches: int, *meters):
        """
        Initializes the ProgressMeter.

        Args:
            prefix (str): Prefix for logging (e.g., "Training Epoch 1").
            num_batches (int): Total number of batches.
            meters (list): List of AverageMeter objects.
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def printt(self, batch: int):
        """
        Prints the progress update.

        Args:
            batch (int): Current batch index.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state: dict, is_best: bool, out_dir: str = '.'):
    """
    Saves model checkpoint.

    Args:
        state (dict): Model state dictionary.
        is_best (bool): Whether this is the best model so far.
        out_dir (str): Directory where to save the checkpoint.
    """
    filename = os.path.join(out_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(out_dir, 'model_best.pth.tar'))


def load_checkpoint(path: str) -> dict:
    """
    Loads a checkpoint file.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    return torch.load(path)