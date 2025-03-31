import os
import json
import sys
from src.evaluation.nms import nms, standard_nms  # Make sure you have this module
from old.load_data_utils import REVERSE_LABELS

class JsonHandler:
    """
    Handles JSON operations for saving model predictions.
    """

    def __init__(self, out_dir: str):
        """
        Initializes the JSON handler.

        Args:
            out_dir (str): Directory where prediction JSON files will be stored.
        """
        self.preds = {}
        self.out_dir = out_dir

    def update_preds(self, match, half, pred_classes, start_frame, scores, time_shifts, frames_per_clip):
        """
        Updates the predictions dictionary.

        Args:
            match (list): List of match names.
            half (list): Corresponding half (1 or 2).
            pred_classes (list): Predicted event classes.
            start_frame (tensor): Start frame indexes.
            scores (tensor): Confidence scores.
            time_shifts (tensor): Predicted time shifts.
            frames_per_clip (int): Number of frames per clip.
        """
        for m, h, c, f, s, t in zip(match, half, pred_classes, start_frame, scores, time_shifts):
            spot = f + t * frames_per_clip  # Adjust event timing
            if m not in self.preds:
                self.preds[m] = {"UrlLocal": m, "predictions": []}

            self.preds[m]["predictions"].append({
                "gameTime": f"{h.item()} - {int(spot / 120):02}:{int((spot % 120) / 2):02}",
                "label": REVERSE_LABELS[int(c)],
                "position": str(int(spot / 2 * 1000)),  # Convert to ms
                "half": str(h.item()),
                "confidence": str(s.item())
            })

    def reset(self):
        """Resets the stored predictions."""
        self.preds = {}

    def save_json(self, epoch, test_split, nms_mode, nms_thresh):
        """
        Saves predictions as JSON files after applying NMS.

        Args:
            epoch (int): Current training epoch.
            test_split (str): Dataset split ("val" or "test").
            nms_mode (str): "standard" or "new".
            nms_thresh (int): NMS threshold.

        Returns:
            str: Path where the predictions were saved.
        """
        predictions_path = os.path.join(self.out_dir, test_split, str(epoch))

        os.makedirs(predictions_path, exist_ok=True)

        if nms_mode == "new":
            preds_after_nms = nms(self.preds, nms_thresh)
        elif nms_mode == "standard":
            preds_after_nms = standard_nms(self.preds, nms_thresh)
        else:
            print("Error: Invalid NMS mode specified. Must be 'standard' or 'new'.")
            sys.exit()

        print(f"Saving predictions JSON at epoch {epoch}...")

        for match, pred in preds_after_nms.items():
            match_dir = os.path.join(predictions_path, match)
            os.makedirs(match_dir, exist_ok=True)

            with open(os.path.join(match_dir, "results_spotting.json"), "w") as outfile:
                json.dump(pred, outfile, indent=4)

        return predictions_path