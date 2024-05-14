""" HW11 - Sparse 3D reconstruction using stereo vision
    
    This Script loads a pair of rectified infrared images, then allows you to select
    up to 30 pixels in each of them to calculate their 3D position using the camera calibration parameters
    and some formulas related with Stereo Vision.

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
                                    description='Select 2D points and generate a basic 3D reconstruction from Stereo Images', 
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

def select_points(image, title='Select points'):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    points = np.asarray(plt.ginput(30, timeout=-1))  # Permite seleccionar 30 puntos
    plt.close(fig)
    return points

def show_and_select_points(left_image, left_points, right_image):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(left_image, cmap='gray')
    axs[0].plot(left_points[:, 0], left_points[:, 1], 'ro', markersize=1.5)
    axs[0].set_title('Left Image with Selected Points')
    axs[1].imshow(right_image, cmap='gray')
    axs[1].set_title('Select points on Right Image')
    right_points = np.asarray(plt.ginput(30, timeout=-1))  # Selección de puntos en la imagen derecha
    plt.close(fig)
    return right_points

def calculate_3d_coordinates(left_pts, right_pts):
    disparities = left_pts[:, 0] - right_pts[:, 0]
    Z = np.where(disparities != 0, (f * B) / disparities, np.inf)
    X = np.where(disparities != 0, (left_pts[:, 0] - cx) * Z / f, 0)
    Y = np.where(disparities != 0, (left_pts[:, 1] - cy) * Z / f, 0)
    return np.column_stack((X, Y, Z))

def run_pipeline():
    """
    """
    # Parse user input
    print("\nInitializing...", end="\r")
    user_input =  parse_user_data()

    global B 
    global f 
    global cx 
    global cy 

    # Loading calibration data
    print("Loading calibration data...", end="\r")
    calibration_data = load_camera_calibration('D:\Rodrigo\Desktop\ITR UDEM\SEM8\VISION_COMPUTACIONAL\TAREA11\calibration-parameters.txt')

    # Extraer los parámetros de calibración
    B = abs(float(calibration_data['baseline']))  # Podemos tomar el valor absoluto por si el signo es incorrecto
    f = float(calibration_data['rectified_fx'])  # Focal length in pixels (assuming fx and fy are the same)
    cx = float(calibration_data['rectified_cx'])  # X-coordinate of the principal point
    cy = float(calibration_data['rectified_cy'])  # Y-coordinate of the principal point
    width = int(calibration_data['rectified_width'])  # Width of the rectified images
    height = int(calibration_data['rectified_height'])  # Height of the rectified images

    # Print calibration data
    print("Calibration data loaded:   ")
    print(f" \u2219 Baseline (B): {B} mm")
    print(f" \u2219 Focal length (f): {f} px")
    print(f" \u2219 Principal point X (cx): {cx} px")
    print(f" \u2219 Principal point Y (cy): {cy} px")
    print(f" \u2219 Image width: {width} px")
    print(f" \u2219 Image height: {height} px")

    l_img = load_image(user_input.left_image)
    r_img = load_image(user_input.right_image)

    visualise_image(l_img,"Left Image")
    visualise_image(r_img,"Right Image")

    close_windows()

    # Selección de puntos en la imagen izquierda
    left_points = select_points(l_img, 'Select points on Left Image')

    # Mostrar puntos en la imagen izquierda y seleccionar puntos en la derecha
    right_points = show_and_select_points(l_img, left_points, r_img)

    coordinates_3d = calculate_3d_coordinates(left_points, right_points) # Calcular
    # coordinates_3d = np.loadtxt('coordinates_3d.csv', delimiter=',', skiprows=1) # Cargar

    # Imprimir las coordenadas calculadas
    print("\nPoints:")
    for i, coord in enumerate(coordinates_3d):
        print(f" \u2219 Point {i+1}: X={coord[0]:.4f}, Y={coord[1]:.4f}, Z={coord[2]:.4f}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates_3d[:, 0], coordinates_3d[:, 2], coordinates_3d[:, 1])
    plt.axis('Equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Reconstruction of Selected Points\n(Camera Axis Reference Orientation)')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates_3d[:, 0], coordinates_3d[:, 2], -coordinates_3d[:, 1])
    plt.axis('Equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Reconstruction of Selected Points\n(Visual Reference Orientation)')
    plt.show()

    # Save coordinates
    save_name = #_coordinates_3d.csv'
    np.savetxt(save_name, coordinates_3d, delimiter=',', header='X,Y,Z', comments='')
    print("3D Coordinates Saved as",save_name)

if __name__ == "__main__":
    run_pipeline()

"""
    References:
"""