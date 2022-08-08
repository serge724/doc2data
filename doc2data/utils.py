# SPDX-FileCopyrightText: 2022-present Sergej Levich <sergej.levich@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities."""

import numpy as np
from PIL import Image


def convert_to_rgb(image, add_white_background=True):
    """Converts a PIL Image from RGBA to RGB.

    White background is added per default to prevent transparent pixels to be
    rendered black.
    """

    if image.mode == "RGBA":
        if add_white_background:
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            white_bg.paste(image, mask=image.split()[3])
            image = white_bg
        else:
            image = image.convert("RGB")

    return image


def load_image(file_path, target_size=None, to_array=True, force_rgb=True):
    """Loads image from file."""

    image = Image.open(file_path)
    if target_size:
        image = image.resize(target_size, resample=None)
    if image.mode == "RGBA" and force_rgb:
        image = convert_to_rgb(
            image, add_white_background=True
        )  # adds white background to RGBA images
    if to_array:
        image = np.array(image)

    return image


def get_pcnt_chars_corrupted(page):
    """Calculates proportion of replacement characters on page."""

    checks = []
    for block in page.get_text("rawdict", flags=0)[
        "blocks"
    ]:  # flags value excludes any images on page
        for line in block["lines"]:
            for span in line["spans"]:
                for char in span["chars"]:
                    checks.append(bool(ord(char["c"]) == 65533))

    pcnt_chars_corrupted = sum(checks) / len(checks) if len(checks) > 0 else None

    return pcnt_chars_corrupted


def normalize_bbox(bounding_box, width, height):
    """Calculates relative coordinates of a bounding box."""

    return [
        bounding_box[0] / width,
        bounding_box[1] / height,
        bounding_box[2] / width,
        bounding_box[3] / height,
    ]


def denormalize_bbox(bounding_box, width, height):
    """Calculates absolute coordinates of a bounding box."""

    return [
        int(bounding_box[0] * width),
        int(bounding_box[1] * height),
        int(bounding_box[2] * width),
        int(bounding_box[3] * height),
    ]
