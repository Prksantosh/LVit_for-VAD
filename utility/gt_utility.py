# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:13:51 2026

@author: USER
"""

from pathlib import Path
import numpy as np


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path(r"C:\Users\USER\Desktop\LVit\data\test")
LABEL_DIR = Path(r"C:\Users\USER\Desktop\LVit\data\test\test_labels")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Define abnormal frame ranges
#
# IMPORTANT:
# These are 1-based frame numbers.
#
# Example:
# (61, 95) means frames 61 through 95 are abnormal.
# ============================================================

ANOMALY_RANGES = {

    "01_0016": [
        (61, 95),
        (140, 165),
    ],

    "01_0029": [
        (25, 50),
    ],

    "12_0151": [
        (100, 130),
        (180, 210),
    ],

}


# ============================================================
# Helper
# ============================================================

def get_frames(video_dir):

    frames = sorted(
        [
            p
            for p in video_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    return frames


# ============================================================
# Create labels
# ============================================================

for video_dir in sorted(TEST_DIR.iterdir()):

    if not video_dir.is_dir():
        continue

    video_name = video_dir.name

    frames = get_frames(video_dir)

    num_frames = len(frames)

    if num_frames == 0:
        print(
            f"[WARNING] No frames found: "
            f"{video_name}"
        )
        continue


    # ----------------------------------------
    # Initially every frame is NORMAL
    # ----------------------------------------

    labels = np.zeros(
        num_frames,
        dtype=np.uint8
    )


    # ----------------------------------------
    # Get abnormal ranges
    # ----------------------------------------

    ranges = ANOMALY_RANGES.get(
        video_name,
        []
    )


    for start_frame, end_frame in ranges:

        # Convert from 1-based frame numbering
        # to Python 0-based array indexing

        start_idx = start_frame - 1

        # end index is exclusive in numpy
        end_idx = end_frame


        # Safety checks
        start_idx = max(
            start_idx,
            0
        )

        end_idx = min(
            end_idx,
            num_frames
        )


        labels[
            start_idx:end_idx
        ] = 1


    # ----------------------------------------
    # Save
    # ----------------------------------------

    output_path = (
        LABEL_DIR
        / f"{video_name}.npy"
    )

    np.save(
        output_path,
        labels
    )


    print(
        f"{video_name}: "
        f"frames={num_frames}, "
        f"normal={np.sum(labels == 0)}, "
        f"anomalous={np.sum(labels == 1)}"
    )

    print(
        f"Saved: {output_path}"
    )


print("\nFinished creating labels.")