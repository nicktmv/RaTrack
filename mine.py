import os
import logging
from csv import DictWriter
from datetime import datetime
from os.path import join
from PIL import Image

import cv2
import numpy as np
from matplotlib import pyplot as plt

from version import __version__


def save_eval_results(seg_met: dict, flow_met: dict) -> None:
    """
    Save the segmentation and flow metrics for a single epoch to a CSV file.
    Args:
        seg_met: A dictionary containing the segmentation metrics for a single epoch.
        flow_met: A dictionary containing the flow metrics for a single epoch.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Ensure timestamp is added last
    seg_met["timestamp"] = timestamp
    flow_met["timestamp"] = timestamp

    seg_met["sw-version"] = __version__
    flow_met["sw-version"] = __version__

    # Define the order of columns explicitly for segmentation metrics
    seg_fieldnames = ["acc", "miou", "sen", "timestamp", "sw-version"]

    # Define the order of columns explicitly for flow metrics
    flow_fieldnames = [
        "rne",
        "50-50 rne",
        "mov_rne",
        "stat_rne",
        "sas",
        "ras",
        "epe",
        "timestamp",
        "sw-version",
    ]

    # Save results to CSV
    folder_results = "./artifacts/eval/"

    # Check if the directory exists, if not, create it
    os.makedirs(folder_results, exist_ok=True)

    # Save the segmentation metrics with the explicit fieldnames
    save_json_list_to_csv(
        [seg_met],
        join(folder_results, "eval-segmentation-metrics.csv"),
        fieldnames=seg_fieldnames,
    )

    # Save the flow metrics with the explicit fieldnames
    save_json_list_to_csv(
        [flow_met],
        join(folder_results, "eval-scene-flow-metrics.csv"),
        fieldnames=flow_fieldnames,
    )


def save_epoch_training_results(
    epoch_number: int, seg_met: dict, flow_met: dict
) -> None:
    # Setting Epoch Number
    seg_met["epoch"] = epoch_number
    flow_met["epoch"] = epoch_number

    # Setting Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    seg_met["timestamp"] = timestamp
    flow_met["timestamp"] = timestamp

    # Setting Software Version
    seg_met["sw-version"] = __version__
    flow_met["sw-version"] = __version__

    # Define the order of columns explicitly for segmentation metrics
    seg_fieldnames = ["epoch", "acc", "miou", "sen", "timestamp", "sw-version"]

    # Define the order of columns explicitly for flow metrics
    flow_fieldnames = [
        "epoch",
        "rne",
        "50-50 rne",
        "mov_rne",
        "stat_rne",
        "sas",
        "ras",
        "epe",
        "timestamp",
        "sw-version",
    ]

    # Save results to CSV
    folder_results = "./artifacts/train/"

    # Check if the directory exists, if not, create it
    os.makedirs(folder_results, exist_ok=True)

    # Save the segmentation metrics with the explicit fieldnames
    save_json_list_to_csv(
        [seg_met],
        join(folder_results, "train-segmentation-metrics.csv"),
        fieldnames=seg_fieldnames,
    )

    # Save the flow metrics with the explicit fieldnames
    save_json_list_to_csv(
        [flow_met],
        join(folder_results, "train-scene-flow-metrics.csv"),
        fieldnames=flow_fieldnames,
    )


def save_json_list_to_csv(
    json_list: list[dict], filename: str, mode: str = "a", fieldnames: list = None
) -> None:
    """Save data to a CSV file.

    Args:
        json_list (list): List of dictionaries containing data to be saved.
        filename (str): Name of the CSV file to save.
        mode (str): Mode to open the CSV file. Default is 'a' (append).
        fieldnames (list): Explicit list of field names for the CSV file.
    """
    if not json_list:
        logging.warning("No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Use the provided fieldnames or deduce them from the JSON list.
    if fieldnames is None:
        fieldnames = list(set().union(*(json_dict.keys() for json_dict in json_list)))

    if not os.path.exists(filename):
        mode = "w"

    with open(filename, mode=mode, newline="", encoding="utf-8") as file:
        dict_writer = DictWriter(file, fieldnames=fieldnames)
        if mode == "w":
            dict_writer.writeheader()
        dict_writer.writerows(json_list)


# Visualisation functions


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def my_obj_centre(obj):
    """Dummy function to calculate the center of the object."""
    return obj.mean(dim=1)


def display_point_with_image_frame(folder_results_vis, index, frame_data):

    cv_image = frame_data.get_image()

    # Load the point cloud image file
    pc_image_path = os.path.join(folder_results_vis, f"seq{index}.png")
    pc_image = Image.open(pc_image_path)  # Open the image file
    pc_image_array = np.array(pc_image)  # Convert the image to an array

    # Flip the point cloud image left to right
    # pc_image_array = np.fliplr(pc_image_array)

    # Combine and display images
    combined_image = combine_images(pc_image_array, cv_image)
    cv2.imshow("Point Cloud and Image", cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Wait for 1 ms to allow GUI to refresh and handle window events

    if cv2.getWindowProperty("Point Cloud and Image", cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()  # Close the window if it has been closed by the user


def combine_images(
    img1, img2, border_color=(255, 255, 255), border_width=10, max_width=1920
):
    """
    This function combines two images side by side with a border in between.
    The images are resized to match in height and adjusted in width to not exceed a maximum width.
    The images are expected to be in the form of numpy arrays with shape (height, width, channels).
    The border color, border width, and maximum width can be customized.

    Args:
        img1 (np.ndarray): First image as a numpy array.
        img2 (np.ndarray): Second image as a numpy array.
        border_color (tuple): RGB color of the border. Default is white.
        border_width (int): Width of the border in pixels. Default is 10.
        max_width (int): Maximum width of the combined image. Default is 1920.

    Returns:
        np.ndarray: Combined image as a numpy array.
    """

    # Check if images have 4 channels and convert to 3 if necessary
    if img1.shape[2] == 4:
        img1 = img1[:, :, :3]
    if img2.shape[2] == 4:
        img2 = img2[:, :, :3]

    # Resize images to match in height and adjust width to not exceed max_width
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate scale to fit both images and border within max_width
    scale_factor = min((max_width - border_width) / (w1 + w2), 1)

    # Resize both images with the same scale factor to maintain aspect ratio
    img1 = cv2.resize(img1, (int(w1 * scale_factor), int(h1 * scale_factor)))
    img2 = cv2.resize(img2, (int(w2 * scale_factor), int(h2 * scale_factor)))

    # Adjust height of the images to match by adding a black border to the smaller image
    if img1.shape[0] > img2.shape[0]:
        delta_h = img1.shape[0] - img2.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        img2 = cv2.copyMakeBorder(
            img2, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    elif img2.shape[0] > img1.shape[0]:
        delta_h = img2.shape[0] - img1.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        img1 = cv2.copyMakeBorder(
            img1, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # Create a vertical border with 3 channels
    border = np.full((img1.shape[0], border_width, 3), border_color, dtype=np.uint8)

    # Concatenate images with a border in between
    combined_image = np.hstack((img1, border, img2))
    return combined_image
