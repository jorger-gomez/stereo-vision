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
# Import standard libraries
import argparse
import cv2
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def parse_user_data() -> tuple[str, str]:
    """
    Parse command-line arguments to obtain paths to stereo image pair.

    Returns:
        tuple[str, str]: Paths to the left and right images.
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
                        help="Path to the right image")
    
    args = parser.parse_args()
    return args

def load_image(filename: str) -> np.ndarray:
    """
    Load an image from the specified file path.

    Args:
        filename (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        SystemExit: If the image file cannot be found or read.
    """
    try:
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError(f"File not found or unsupported format: {filename}")
        return img
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

def resize_image(img: np.ndarray, scale: int = 100) -> np.ndarray:
    """
    Resize the image by a specified percentage scale.

    Args:
        img (np.ndarray): Image to resize.
        scale (int): Percentage to scale the image (default 100%).

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

def close_windows() -> None:
    """
    Close and destroy all OpenCV windows.
    
    Returns:
        None
    """
    cv2.destroyAllWindows()

def load_camera_calibration(path: str) -> dict:
    """
    Load camera calibration parameters from a file.

    Args:
        path (str): Path to the calibration data file.

    Returns:
        dict: Calibration data parameters.

    Raises:
        SystemExit: If the calibration file cannot be found or read.
    """
    try:
        with open(path, 'r') as file:
            calibration_data = json.load(file)
        return calibration_data
    except FileNotFoundError:
        print("Calibration file not found.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error decoding the calibration file.", file=sys.stderr)
        sys.exit(1)

def select_points(image: np.ndarray, title: str = 'Select points') -> np.ndarray:
    """
    Interactive selection of points on an image displayed.

    Args:
        image (np.ndarray): Image on which to select points.
        title (str): Title of the window showing the image.

    Returns:
        np.ndarray: Array containing selected points.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    points = np.asarray(plt.ginput(30, timeout=-1))  # Allow selection of max 30 points 
    plt.close(fig)
    return points

def show_and_select_points(left_image: np.ndarray, left_points: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    """
    Display left and right images side-by-side, show selected points on the left image, and allow point selection on the right image.

    Args:
        left_image (np.ndarray): The left image of the stereo pair.
        left_points (np.ndarray): Points selected on the left image.
        right_image (np.ndarray): The right image of the stereo pair.

    Returns:
        np.ndarray: Points selected on the right image.

    Raises:
        ValueError: If the input data is not valid.
    """
    try:
        # Validate input types and content
        if not isinstance(left_image, np.ndarray) or not isinstance(right_image, np.ndarray):
            raise ValueError("Input images must be numpy arrays.")
        if left_points.ndim != 2 or left_points.shape[1] != 2:
            raise ValueError("Left points must be a two-dimensional array with two columns.")

        # Set up the figure and axes for plotting
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(left_image, cmap='gray')
        axs[0].plot(left_points[:, 0], left_points[:, 1], 'ro', markersize=1.5)
        axs[0].set_title('Left Image with Selected Points')
        axs[1].imshow(right_image, cmap='gray')
        axs[1].set_title('Select points on Right Image')

        # Interactive point selection on the right image
        right_points = np.asarray(plt.ginput(30, timeout=-1))  # Allows selecting max 30 points
        
        # Close the plotting figure
        plt.close(fig)
        
        if right_points.size == 0:
            raise ValueError("No points were selected on the right image.")
        return right_points

    except Exception as e:
        print(f"Error during point selection: {e}", file=sys.stderr)
        sys.exit(1)

def calculate_3d_coordinates(left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
    """
    Calculate 3D coordinates based on disparities between corresponding points in stereo images.

    Args:
        left_pts (np.ndarray): Points from the left image.
        right_pts (np.ndarray): Corresponding points from the right image.

    Returns:
        np.ndarray: 3D coordinates of the points.

    Raises:
        Exception: If there is a failure in calculation.
    """
    try:
        disparities = left_pts[:, 0] - right_pts[:, 0]
        Z = np.where(disparities != 0, (f * B) / disparities, np.inf)
        X = np.where(disparities != 0, (left_pts[:, 0] - cx) * Z / f, 0)
        Y = np.where(disparities != 0, (left_pts[:, 1] - cy) * Z / f, 0)
        return np.column_stack((X, Y, Z))
    except Exception as e:
        print(f"Failed to calculate 3D coordinates: {e}", file=sys.stderr)
        sys.exit(1)

def run_pipeline() -> None:
    """
    Main function to run the stereo vision pipeline, including loading images, calibration, and 3D reconstruction.
    """
    # Initialize and parse user input
    print("\nInitializing...", end="\r")
    user_input =  parse_user_data()

    # Declare global variables for calibration parameters
    global B, f, cx, cy  

    # Load calibration data
    print("Loading calibration data...", end="\r")
    calibration_data = load_camera_calibration('calibration-parameters.txt')

    # Extraer los parámetros de calibración
    B = abs(float(calibration_data['baseline']))  # Baseline, using absolute value to correct sign
    f = float(calibration_data['rectified_fx'])  # Focal length in pixels (assuming fx and fy are the same)
    cx = float(calibration_data['rectified_cx'])  # X-coordinate of the principal point
    cy = float(calibration_data['rectified_cy'])  # Y-coordinate of the principal point

    # Print calibration data
    print("Calibration data loaded:   ")
    print(f" \u2219 Baseline (B): {B} mm")
    print(f" \u2219 Focal length (f): {f} px")
    print(f" \u2219 Principal point X (cx): {cx} px")
    print(f" \u2219 Principal point Y (cy): {cy} px")
    print(f" \u2219 Image width: {int(calibration_data['rectified_width'])} px")
    print(f" \u2219 Image height: {int(calibration_data['rectified_height'])} px\n")

    # Load images
    l_img = load_image(user_input.left_image)
    r_img = load_image(user_input.right_image)

    # Display images for verification
    visualise_image(l_img,"Left Image")
    visualise_image(r_img,"Right Image")

    # Close OpenCV windows
    close_windows()

    # Select points on the left image
    left_points = select_points(l_img, 'Select points on Left Image')

    # Show points on the left image and select points on the right image
    right_points = show_and_select_points(l_img, left_points, r_img)

    # Calculate 3D coordinates from selected points
    coordinates_3d = calculate_3d_coordinates(left_points, right_points)
    # coordinates_3d = np.loadtxt('#_coordinates_3d.csv', delimiter=',', skiprows=1) # LOAD SAVED DATA

    # Display the calculated 3D points
    print("Points:")
    for i, coord in enumerate(coordinates_3d):
        print(f" \u2219 Point {i+1:2d}: ({coord[0]:.5f}mm, {coord[1]:.5f}mm, {coord[2]:.5f}mm)")

    # Plot 3D points for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates_3d[:, 0], coordinates_3d[:, 2], -coordinates_3d[:, 1], c='purple', edgecolor='black', s=50, alpha=0.6)
    plt.axis('Equal')
    ax.set_xlabel('X (mm)', fontweight='bold')
    ax.set_ylabel('Z (mm)', fontweight='bold')
    ax.set_zlabel('Y (mm)', fontweight='bold')
    ax.set_title('3D Reconstruction of Selected Points', fontsize=14, fontweight='bold')
    ax.view_init(elev=25, azim=90)
    ax.grid(False)

    plt.show()

    # Save coordinates. Comment This section to avoid saving data.
    save_name = 'book_coordinates_3d.csv' # Change value to change output file name
    np.savetxt(save_name, coordinates_3d, delimiter=',', header='X,Y,Z', comments='')
    print("\n3D Coordinates saved as",save_name)

if __name__ == "__main__":
    run_pipeline()

"""
    References:
    [1]“OpenCV: Epipolar Geometry.” Accessed: May 15, 2024. [Online]. 
        Available: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

    [2]“OpenCV: Pose Estimation.” Accessed: May 15, 2024. [Online]. 
        Available: https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html  
"""