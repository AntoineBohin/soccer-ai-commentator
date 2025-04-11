import os
import json
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader

import torch
import torch._dynamo

from argus import load_model
from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
)

from src.datasets.annotations import get_videos_data, get_videos_sampling_weights
from src.utils import load_weights_from_pretrain, get_best_model_path, get_lr
from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.datasets.datasets import TrainActionDataset, ValActionDataset
from src.data_processing.augmentations import get_train_augmentations
from src.training.metrics import AveragePrecision, Accuracy
from src.data_processing.data_loaders.random_seek import RandomSeekDataLoader
from src.datasets.target import MaxWindowTargetsProcessor
from src.models.customcnn import BallActionModel
from src.training.ema import ModelEma, EmaCheckpoint
from src.data_processing.frames import get_frames_processor
from src.datasets import constants
from src.data_processing.mixup import TimmMixup

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str) # Get all the
    return parser.parse_args()


def train_action(config: dict, save_dir: Path):
    argus_params = config["argus_params"]
    model = BallActionModel(argus_params)
    if "pretrained" in model.params["nn_module"][1]:
        model.params["nn_module"][1]["pretrained"] = False

    if "pretrain_action_experiment" in config and config["pretrain_action_experiment"]:
        pretrain_dir = constants.experiments_dir / config["pretrain_action_experiment"]
        pretrain_model_path = get_best_model_path(pretrain_dir)
        print(f"Load pretrain model: {pretrain_model_path}")
        pretrain_model = load_model(pretrain_model_path, device=argus_params["device"])
        load_weights_from_pretrain(model.nn_module, pretrain_model.nn_module)
        del pretrain_model

    # Data Augmentation (OPTIONAL) 
    # TO DO
    augmentations = get_train_augmentations(config["image_size"])
    model.augmentations = augmentations

    # Data MixUp (OPTIONAL) 
    # TO DO
    if "mixup_params" in config:
        model.mixup = TimmMixup(**config["mixup_params"])

    # Processor for Max Pooling when overlapping targets/predictions (data.target.py)
    targets_processor = MaxWindowTargetsProcessor(
        window_size=config["max_targets_window_size"]
    )
    # Frame Processor (frames.py) to get normalization/padding of the frames
    frames_processor = get_frames_processor(*argus_params["frames_processor"])
    # Indexes generator (indexes.py) to get the frame indexes of an image (stack size = 15)
    indexes_generator = StackIndexesGenerator(
        argus_params["frame_stack_size"],
        argus_params["frame_stack_step"],
    )
    # Index shaker to shake frame indexes randomly
    frame_index_shaker = FrameIndexShaker(**config["frame_index_shaker"])

    # EMA decay for model weights (robustness to noise/overfitting, stability) - (ex: 0.9995)
    # TO DO
    print("EMA decay:", config["ema_decay"])
    model.model_ema = ModelEma(model.nn_module, decay=config["ema_decay"])

    # Compile model for performance optimization if specified (OPTIONAL)
    if "torch_compile" in config:
        print("torch.compile:", config["torch_compile"])
        torch._dynamo.reset()
        model.nn_module = torch.compile(model.nn_module, **config["torch_compile"])

    # Training device (GPU/CPU)
    device = torch.device(argus_params["device"][0])

    # Initizalisation of the Train Dataset (data.datasets.py)
    # Loading training data - annotations, frames index... (data.annotations.py)
    # For the chosen train games
    train_data = get_videos_data(constants.train_games)
    # Get sampling weights depending on actions (data.annotations.py)
    videos_sampling_weights = get_videos_sampling_weights(
        train_data, **config["train_sampling_weights"],
    )
    train_dataset = TrainActionDataset(
        train_data,
        constants.classes,
        indexes_generator=indexes_generator,
        epoch_size=config["train_epoch_size"],
        videos_sampling_weights=videos_sampling_weights,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
        frame_index_shaker=frame_index_shaker,
    )
    print(f"Train dataset len {len(train_dataset)}")
    # Initizalisation of the Validation Dataset (data.datasets.py)
    # Loading validation data - annotations, frames index... (data.annotations.py)
    # With optional empty actions for full video coverage
    val_data = get_videos_data(constants.val_games, add_empty_actions=True)
    val_dataset = ValActionDataset(
        val_data,
        constants.classes,
        indexes_generator=indexes_generator,
        target_process_fn=targets_processor,
        frames_process_fn=frames_processor,
    )
    print(f"Val dataset len {len(val_dataset)}")

    # Training/Validation data loaders - randomly (data_loaders.random_seek.py)
    train_loader = RandomSeekDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_nvdec_workers=config["num_nvdec_workers"],
        num_opencv_workers=config["num_opencv_workers"],
        gpu_id=device.index,
    )
    val_loader = RandomSeekDataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_nvdec_workers=config["num_nvdec_workers"],
        num_opencv_workers=0,
        gpu_id=device.index,
    )
    # Training loop over different training stages (warmup, full training - ex [4, 20] epochs)
    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        # Callbacks for logging
        callbacks = [
            LoggingToFile(save_dir / "log.txt", append=True),
            LoggingToCSV(save_dir / "log.csv", append=True),
        ]

        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "warmup":
            # Warmup stage with linear learning rate increase
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True),
            ]
            model.fit(train_loader,
                      num_epochs=num_epochs,
                      callbacks=callbacks)

        elif stage == "train":
            # Full training stage with learning rate annealing and checkpointing
            checkpoint_format = "model-{epoch:03d}-{val_average_precision:.6f}.pth"
            callbacks += [
                EmaCheckpoint(save_dir, file_format=checkpoint_format, max_saves=num_epochs),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True,
                ),
            ]
            
            # Define evaluation metrics (metrics.py)
            metrics = [
                AveragePrecision(constants.classes),
                Accuracy(constants.classes, threshold=config["metric_accuracy_threshold"]),
            ]

            model.fit(train_loader,
                      val_loader=val_loader,
                      num_epochs=num_epochs,
                      callbacks=callbacks,
                      metrics=metrics)

    # Stop worker processes for data loading
    train_loader.stop_workers()
    val_loader.stop_workers()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = parse_arguments()
    print("Experiment:", args.experiment)

    config_path = constants.configs_dir / f"{args.experiment}.py"
    if not config_path.exists():
        raise RuntimeError(f"Config '{config_path}' is not exists")

    config = SourceFileLoader(args.experiment, str(config_path)).load_module().config
    print("Experiment config:")
    pprint(config, sort_dicts=False)

    experiments_dir = constants.experiments_dir / args.experiment
    print("Experiment dir:", experiments_dir)
    if not experiments_dir.exists():
        experiments_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{experiments_dir}' already exists.")

    with open(experiments_dir / "train.py", "w") as outfile:
        outfile.write(open(__file__).read())

    with open(experiments_dir / "config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    train_action(config, experiments_dir)