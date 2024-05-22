import os
import sys
import cv2
import numpy as np

WINDOW_NAME = 'Preview Window'
image_converted = False
cv2.namedWindow(WINDOW_NAME)
clicks = []
#
#creates a rectangle image from a warped input image
# use the following input to run programm: image-extractor.py <input_file> <output_destination> <width> <height>
# then select the corners of the image part that should be transformed
# click ESC to discard the points and start new
# click S to then save the image
#

try: #chatgpt
    # Check for the correct number of arguments and shows what the input should look like
    if len(sys.argv) != 5:
        raise ValueError("Usage: image-extractor.py <input_file> <output_destination> <width> <height>")

    input_file = sys.argv[1]
    output_destination = sys.argv[2]
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    # Check if input file exists and is a .jpg file
    if not os.path.isfile(input_file) or not input_file.lower().endswith('.jpg'):
        raise ValueError("Input file must be a valid .jpg file")

    # Check if the output destination is a valid path
    output_dir = os.path.dirname(output_destination)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError("Output destination must be a valid path")

    # Check if width and height are positive integers
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")

    # Read the image
    img = cv2.imread(input_file)
    if img is None:
        raise ValueError("Failed to read the input image file")
    original_img = img.copy()

except ValueError as e:
    print(e)
    sys.exit(1)

#orders points so that points can be clicked on in random order, done with chagpt
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

#https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def convert_image():
    global img, clicks, image_converted, straightened_image
    pts_src = np.array(clicks, dtype=np.float32)
    ordered_pts = order_points(pts_src)

    # Define the destination points for the transformed image
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered_pts, pts_dst)
    straightened_image = cv2.warpPerspective(original_img, matrix, (int(width), int(height)))
    # use original_img so selection points aren't on transformed image
    img = cv2.warpPerspective(original_img, matrix, (int(width), int(height)))
    image_converted = True

def mouse_callback(event, x, y, flags, param):
    global img, clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        clicks.append((x, y))
        if len(clicks) == 4:
            convert_image()
            print(clicks)
        cv2.imshow(WINDOW_NAME, img)


cv2.imshow(WINDOW_NAME, img)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and image_converted == True: #checks if image is converted and saves image to intended folder
        cv2.imwrite(f'{output_destination}\\straightened_{os.path.basename(input_file)}', straightened_image)
        sys.exit(0)
    elif key == 27:
        clicks.clear()
        img = original_img.copy()
        cv2.imshow(WINDOW_NAME, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
