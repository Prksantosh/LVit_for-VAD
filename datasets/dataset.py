from __future__ import annotations

import os
import random
from pathlib import Path
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Sequence,
    Union,
)

import numpy as np
from PIL import Image

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    random_split,
)

from torchvision import transforms


# ============================================================
# 1. Supported image formats
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. Reproducibility utility
# ============================================================

def seed_everything(
    seed: int = 42
):
    """
    Set random seeds for reproducible
    train/validation splitting and training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Natural sorting
# ============================================================

def natural_key(
    path: Union[str, Path]
):
    """
    Natural sorting for filenames such as:

        1.png
        2.png
        10.png

    instead of lexical ordering:

        1.png
        10.png
        2.png
    """

    import re

    path = str(path)

    return [
        int(text)
        if text.isdigit()
        else text.lower()

        for text in re.split(
            r"(\d+)",
            path
        )
    ]


# ============================================================
# 4. Discover image frames
# ============================================================

def find_image_files(
    root_dir: Union[str, Path],
    recursive: bool = True
) -> List[Path]:
    """
    Find all image files under root_dir.
    """

    root_dir = Path(root_dir)

    if not root_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{root_dir}"
        )

    if recursive:

        files = [

            path

            for path in root_dir.rglob("*")

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

    else:

        files = [

            path

            for path in root_dir.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

    files = sorted(
        files,
        key=natural_key
    )

    return files


# ============================================================
# 5. Image transformation
# ============================================================

def build_frame_transform(
    image_size: Tuple[int, int] = (
        256,
        256
    ),
):
    """
    Transformation used for reconstruction training.

    PIL RGB image
        ↓
    resize
        ↓
    tensor
        ↓
    [0, 1]

    IMPORTANT:
    No ImageNet normalization is applied because
    the reconstruction head uses Sigmoid and
    produces images in [0, 1].
    """

    transform = transforms.Compose(

        [

            transforms.Resize(
                image_size,
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
                antialias=True
            ),

            transforms.ToTensor(),
        ]
    )

    return transform


# ============================================================
# 6. Basic frame reconstruction dataset
# ============================================================

class FrameReconstructionDataset(Dataset):
    """
    Generic reconstruction dataset.

    Each sample:

        {
            "input":  image,
            "target": image,
            "path":   frame path
        }

    Tensor shape:

        [3, H, W]

    Pixel range:

        [0, 1]
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        image_size: Tuple[int, int] = (
            256,
            256
        ),
        transform=None,
        recursive: bool = True,
        stride: int = 1,
        return_path: bool = True,
    ):
        super().__init__()

        self.root_dir = Path(
            root_dir
        )

        self.image_size = (
            image_size
        )

        self.return_path = (
            return_path
        )

        if stride < 1:

            raise ValueError(
                "stride must be >= 1"
            )

        self.stride = stride

        # ------------------------------------
        # Discover frames
        # ------------------------------------

        all_frames = find_image_files(
            self.root_dir,
            recursive=recursive
        )

        if len(all_frames) == 0:

            raise RuntimeError(
                f"No image frames were found in "
                f"{self.root_dir}"
            )

        # Apply temporal/frame stride

        self.frames = all_frames[
            ::stride
        ]

        # ------------------------------------
        # Transform
        # ------------------------------------

        if transform is None:

            self.transform = (
                build_frame_transform(
                    image_size
                )
            )

        else:

            self.transform = transform

        print(
            f"[FrameReconstructionDataset] "
            f"Loaded {len(self.frames)} frames "
            f"from {self.root_dir}"
        )

    def __len__(
        self
    ):

        return len(
            self.frames
        )

    def _load_image(
        self,
        path: Path
    ):

        with Image.open(path) as image:

            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        return image

    def __getitem__(
        self,
        index
    ):

        frame_path = (
            self.frames[index]
        )

        image = self._load_image(
            frame_path
        )

        # Reconstruction target
        # intentionally equals input.

        target = image.clone()

        sample = {

            "input": image,

            "target": target,
        }

        if self.return_path:

            sample["path"] = str(
                frame_path
            )

        return sample


# ============================================================
# 7. Video-aware reconstruction dataset
# ============================================================

class VideoFrameReconstructionDataset(
    Dataset
):
    """
    Video-aware frame reconstruction dataset.

    Recommended when the dataset structure is:

        train/
            video_01/
                frame_001.png
                frame_002.png
                ...
            video_02/
                ...
            video_03/
                ...

    This retains video identity and frame index.

    Each sample:

        {
            input,
            target,
            video_name,
            frame_index,
            path
        }
    """

    def __init__(
        self,
        root_dir,
        image_size=(256, 256),
        stride=1,
        transform=None,
        return_metadata=True,
    ):
        super().__init__()

        self.root_dir = Path(
            root_dir
        )

        self.stride = stride

        self.return_metadata = (
            return_metadata
        )

        if transform is None:

            self.transform = (
                build_frame_transform(
                    image_size
                )
            )

        else:

            self.transform = transform

        # ------------------------------------
        # Discover video folders
        # ------------------------------------

        video_dirs = sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.is_dir()
            ],
            key=natural_key
        )

        self.samples = []

        # ------------------------------------
        # Case 1: videos stored in subfolders
        # ------------------------------------

        if len(video_dirs) > 0:

            for video_dir in video_dirs:

                frames = (
                    find_image_files(
                        video_dir,
                        recursive=False
                    )
                )

                frames = frames[
                    ::stride
                ]

                for frame_index, path in enumerate(
                    frames
                ):

                    self.samples.append(
                        {
                            "path": path,
                            "video_name": (
                                video_dir.name
                            ),
                            "frame_index": (
                                frame_index
                            ),
                        }
                    )

        # ------------------------------------
        # Case 2: all frames directly in root
        # ------------------------------------

        else:

            frames = find_image_files(
                self.root_dir,
                recursive=False
            )

            frames = frames[
                ::stride
            ]

            for frame_index, path in enumerate(
                frames
            ):

                self.samples.append(
                    {
                        "path": path,
                        "video_name": (
                            self.root_dir.name
                        ),
                        "frame_index": (
                            frame_index
                        ),
                    }
                )

        if len(self.samples) == 0:

            raise RuntimeError(
                f"No frames found in "
                f"{self.root_dir}"
            )

        print(
            f"[VideoFrameDataset] "
            f"{len(self.samples)} frames found."
        )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        info = self.samples[
            index
        ]

        path = info[
            "path"
        ]

        with Image.open(path) as image:

            image = image.convert(
                "RGB"
            )

            frame = self.transform(
                image
            )

        sample = {

            "input": frame,

            "target": frame.clone(),
        }

        if self.return_metadata:

            sample.update(
                {
                    "video_name":
                        info["video_name"],

                    "frame_index":
                        info["frame_index"],

                    "path":
                        str(path),
                }
            )

        return sample


# ============================================================
# 8. Dataset with anomaly labels
# ============================================================

class LabeledFrameDataset(Dataset):
    """
    Generic test dataset for anomaly detection.

    image_paths:
        list of frame paths

    labels:
        frame-level binary labels

        0 -> normal
        1 -> anomalous

    This should normally be used for evaluation,
    NOT for training the unsupervised model.
    """

    def __init__(
        self,
        image_paths: Sequence[
            Union[str, Path]
        ],
        labels: Sequence[int],
        image_size=(256, 256),
        transform=None
    ):
        super().__init__()

        if (
            len(image_paths)
            != len(labels)
        ):

            raise ValueError(
                "Number of image paths and "
                "labels must be identical."
            )

        self.image_paths = [
            Path(path)
            for path in image_paths
        ]

        self.labels = np.asarray(
            labels,
            dtype=np.int64
        )

        if transform is None:

            self.transform = (
                build_frame_transform(
                    image_size
                )
            )

        else:

            self.transform = transform

    def __len__(
        self
    ):

        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index
    ):

        path = (
            self.image_paths[index]
        )

        label = int(
            self.labels[index]
        )

        with Image.open(path) as image:

            image = image.convert(
                "RGB"
            )

            frame = self.transform(
                image
            )

        return {

            "input": frame,

            "target": frame.clone(),

            "label": torch.tensor(
                label,
                dtype=torch.long
            ),

            "path": str(path),
        }


# ============================================================
# 9. Train / validation split
# ============================================================

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Randomly split a reconstruction dataset.

    Default:
        80% train
        20% validation
    """

    if not (
        0.0
        < train_ratio
        < 1.0
    ):

        raise ValueError(
            "train_ratio must be "
            "between 0 and 1."
        )

    total_size = len(
        dataset
    )

    train_size = int(
        total_size
        * train_ratio
    )

    val_size = (
        total_size
        - train_size
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    train_dataset, val_dataset = (
        random_split(

            dataset,

            [
                train_size,
                val_size
            ],

            generator=generator
        )
    )

    print(
        f"Total frames : {total_size}"
    )

    print(
        f"Train frames : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val frames   : "
        f"{len(val_dataset)}"
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# 10. Preferred video-level train/val split
# ============================================================

def split_video_directories(
    root_dir,
    train_ratio=0.8,
    seed=42
):
    """
    Split by VIDEO rather than randomly mixing frames.

    This is preferable when each video has its own folder,
    because adjacent frames from one video should not be
    simultaneously present in training and validation.

    Returns:
        train_video_dirs,
        val_video_dirs
    """

    root_dir = Path(
        root_dir
    )

    video_dirs = sorted(
        [
            path
            for path in root_dir.iterdir()
            if path.is_dir()
        ],
        key=natural_key
    )

    if len(video_dirs) == 0:

        raise RuntimeError(
            "No video directories found."
        )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        video_dirs
    )

    split_index = int(
        len(video_dirs)
        * train_ratio
    )

    train_dirs = video_dirs[
        :split_index
    ]

    val_dirs = video_dirs[
        split_index:
    ]

    return (
        train_dirs,
        val_dirs
    )


# ============================================================
# 11. Dataset from explicitly selected video folders
# ============================================================

class SelectedVideoDataset(Dataset):
    """
    Construct reconstruction dataset from a known
    set of video folders.

    Useful for video-level train/validation splitting.
    """

    def __init__(
        self,
        video_dirs: Sequence[
            Union[str, Path]
        ],
        image_size=(256, 256),
        stride=1,
        transform=None
    ):
        super().__init__()

        self.video_dirs = [
            Path(path)
            for path in video_dirs
        ]

        if transform is None:

            self.transform = (
                build_frame_transform(
                    image_size
                )
            )

        else:

            self.transform = (
                transform
            )

        self.samples = []

        for video_dir in (
            self.video_dirs
        ):

            frames = (
                find_image_files(
                    video_dir,
                    recursive=False
                )
            )

            frames = frames[
                ::stride
            ]

            for frame_index, path in enumerate(
                frames
            ):

                self.samples.append(
                    (
                        path,
                        video_dir.name,
                        frame_index
                    )
                )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        (
            path,
            video_name,
            frame_index

        ) = self.samples[
            index
        ]

        with Image.open(path) as image:

            image = image.convert(
                "RGB"
            )

            frame = self.transform(
                image
            )

        return {

            "input": frame,

            "target": frame.clone(),

            "video_name":
                video_name,

            "frame_index":
                frame_index,

            "path":
                str(path),
        }


# ============================================================
# 12. DataLoader builder
# ============================================================

def create_dataloader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
):
    """
    Standard DataLoader factory.
    """

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=pin_memory,

        drop_last=drop_last,

        persistent_workers=(
            num_workers > 0
        ),
    )

    return loader


# ============================================================
# 13. Train / Validation DataLoader builder
# ============================================================

def create_train_val_loaders(
    root_dir,
    image_size=(256, 256),
    batch_size=2,
    train_ratio=0.8,
    stride=1,
    num_workers=4,
    seed=42,
):
    """
    Simple training utility.

    Suitable when frame-level random splitting is intended.
    """

    seed_everything(
        seed
    )

    dataset = (
        VideoFrameReconstructionDataset(

            root_dir=root_dir,

            image_size=image_size,

            stride=stride,

            return_metadata=True
        )
    )

    (
        train_dataset,
        val_dataset

    ) = split_dataset(

        dataset,

        train_ratio=train_ratio,

        seed=seed
    )

    train_loader = (
        create_dataloader(

            train_dataset,

            batch_size=batch_size,

            shuffle=True,

            num_workers=num_workers,

            pin_memory=True,

            drop_last=True
        )
    )

    val_loader = (
        create_dataloader(

            val_dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=num_workers,

            pin_memory=True,

            drop_last=False
        )
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# 14. Recommended video-level loader builder
# ============================================================

def create_video_level_train_val_loaders(
    root_dir,
    image_size=(256, 256),
    batch_size=2,
    train_ratio=0.8,
    stride=1,
    num_workers=4,
    seed=42,
):
    """
    RECOMMENDED split for surveillance video data.

    Entire videos are allocated either to train
    OR validation.

    This avoids neighbouring frames from the same
    video leaking into both sets.
    """

    seed_everything(
        seed
    )

    (
        train_video_dirs,
        val_video_dirs

    ) = split_video_directories(

        root_dir=root_dir,

        train_ratio=train_ratio,

        seed=seed
    )

    train_dataset = (
        SelectedVideoDataset(

            video_dirs=train_video_dirs,

            image_size=image_size,

            stride=stride
        )
    )

    val_dataset = (
        SelectedVideoDataset(

            video_dirs=val_video_dirs,

            image_size=image_size,

            stride=stride
        )
    )

    train_loader = (
        create_dataloader(

            train_dataset,

            batch_size=batch_size,

            shuffle=True,

            num_workers=num_workers,

            pin_memory=True,

            drop_last=True
        )
    )

    val_loader = (
        create_dataloader(

            val_dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=num_workers,

            pin_memory=True,

            drop_last=False
        )
    )

    print(
        "\nDataset summary"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Training videos : "
        f"{len(train_video_dirs)}"
    )

    print(
        f"Validation videos: "
        f"{len(val_video_dirs)}"
    )

    print(
        f"Training frames : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation frames: "
        f"{len(val_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset
    )


# ============================================================
# 15. Dataset sanity check
# ============================================================

def inspect_dataset(
    dataset,
    num_samples=3
):
    """
    Print dataset statistics and verify data range.
    """

    print(
        "\n=================================="
    )

    print(
        "Dataset Sanity Check"
    )

    print(
        "=================================="
    )

    print(
        f"Dataset length: "
        f"{len(dataset)}"
    )

    num_samples = min(
        num_samples,
        len(dataset)
    )

    for i in range(
        num_samples
    ):

        sample = dataset[i]

        image = sample[
            "input"
        ]

        target = sample[
            "target"
        ]

        print(
            f"\nSample {i}"
        )

        print(
            "Input shape :",
            tuple(image.shape)
        )

        print(
            "Target shape:",
            tuple(target.shape)
        )

        print(
            "Minimum:",
            float(image.min())
        )

        print(
            "Maximum:",
            float(image.max())
        )

        print(
            "Mean:",
            float(image.mean())
        )

        if "path" in sample:

            print(
                "Path:",
                sample["path"]
            )


# ============================================================
# 16. Example
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        "./dataset/train"
    )

    # ----------------------------------------
    # Recommended approach
    # ----------------------------------------

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset

    ) = (
        create_video_level_train_val_loaders(

            root_dir=DATASET_PATH,

            image_size=(
                256,
                256
            ),

            batch_size=2,

            train_ratio=0.8,

            stride=1,

            num_workers=4,

            seed=42
        )
    )

    inspect_dataset(
        train_dataset
    )

    # ----------------------------------------
    # Verify DataLoader
    # ----------------------------------------

    batch = next(
        iter(train_loader)
    )

    inputs = batch[
        "input"
    ]

    targets = batch[
        "target"
    ]

    print(
        "\n=================================="
    )

    print(
        "DataLoader Test"
    )

    print(
        "=================================="
    )

    print(
        "Input batch:",
        inputs.shape
    )

    print(
        "Target batch:",
        targets.shape
    )

    print(
        "Input range:",
        inputs.min().item(),
        inputs.max().item()
    )