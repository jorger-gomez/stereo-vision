""" HW11 - Sparse 3D reconstruction using stereo vision
    
    description

    Authors: Jorge Rodrigo Gómez Mayo 
    Contact: jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
    First created on Saturday 11 May 2024

    Usage Examples:
        py .\stereo-vision.py -l_img .\rectified-images\left_infrared_image.png -r_img .\rectified-images\right_infrared_image.png
"""
# Import std libs
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


def parse_user_data() -> tuple[str, str]:
    """
    Parse the command-line arguments provided by the user.

    Returns:
        tuple[str, str]: A tuple containing the path to the object image and the input image.
    """
    parser = argparse.ArgumentParser(prog='HW11 - Sparse 3D reconstruction using stereo vision',
                                    description='txt', 
                                    epilog='JRGM - 2024')
    parser.add_argument('-l_img',
                        '--left_image',
                        type=str,
                        required=True,
                        help="Path to the left image")
    parser.add_argument('-r_img',
                        '--right_image',
                        type=str,
                        required=True,
                        help="Path to the left image")
    
    args = parser.parse_args()
    return args

def load_image(filename: str) -> [np.ndarray]:
    """
    """
    img = cv2.imread(filename)
    return img

def resize_image(img: np.ndarray, scale=100) -> np.ndarray:
    """
    Resize the image to a specified scale for better visualization.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(img.shape[1] * (scale/100))
    height = int(img.shape[0] * (scale/100))
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def visualise_image(img: np.ndarray, title: str, scale: int = 100, picture: bool = True) -> None:
    """
    Display an image in a window with a specified title and scale.

    Args:
        img (np.ndarray): The image to display.
        title (str): The title of the window.
        scale (int): Percentage to scale the image for display.
        picture (bool): Flag to wait for a key press if true.

    Returns:
        None
    """
    resized = resize_image(img, scale)
    cv2.imshow(title, resized)
    if picture:
        cv2.waitKey(0)
    return None

def close_windows():
    """
    Close & destroy OpenCV windows

    The Function closes and destroy all cv2 windows that are open.
    """
    cv2.destroyAllWindows()

def load_camera_calibration(path):
    # Leer y parsear el archivo de calibración
    with open(path, 'r') as file:
        calibration_data = json.load(file)
    return calibration_data

def run_pipeline():
    """
    """
    # Parse user input
    print("Initializing...", end="\r")
    user_input =  parse_user_data()

    # Loading calibration data
    print("Loading calibration data...", end="\r")
    calibration_data = load_camera_calibration('D:\Rodrigo\Desktop\ITR UDEM\SEM8\VISION_COMPUTACIONAL\TAREA11\calibration-parameters.txt')

    # Extraer los parámetros de calibración
    B = float(calibration_data['baseline'])  # Podemos tomar el valor absoluto por si el signo es incorrecto
    f = float(calibration_data['rectified_fx'])  # Focal length in pixels (assuming fx and fy are the same)
    cx = float(calibration_data['rectified_cx'])  # X-coordinate of the principal point
    cy = float(calibration_data['rectified_cy'])  # Y-coordinate of the principal point
    width = int(calibration_data['rectified_width'])  # Width of the rectified images
    height = int(calibration_data['rectified_height'])  # Height of the rectified images

    # Print calibration data
    print("Calibration data loaded:   ")
    print(f"Baseline (B): {B} mm")
    print(f"Focal length (f): {f} px")
    print(f"Principal point X (cx): {cx} px")
    print(f"Principal point Y (cy): {cy} px")
    print(f"Image width: {width} px")
    print(f"Image height: {height} px")

    l_img = load_image(user_input.left_image)
    r_img = load_image(user_input.right_image)

    visualise_image(l_img,"Left Image")
    visualise_image(r_img,"Right Image")

    close_windows()


if __name__ == "__main__":
    run_pipeline()

"""
    References:
"""