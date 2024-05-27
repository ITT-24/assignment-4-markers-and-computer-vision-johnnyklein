import os
import time

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys

from pyglet import clock

video_id = 1
gaussian_kernel = 5
cutoff = 120  #seems to work well in my room with moderate light and in low light
score_p1 = 0
score_p2 = 0
game_started = False
YELLOW = (255, 255, 0, 255)
ORANGE = (255, 140, 0, 255)
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])
# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#ball speed and size set in relation to webcam resolution, my webcam is quite low 640x400 but hope it works with FullHD
BALL_RADIUS = width//60
SLOW = width//200
MEDIUM = width//150
FAST = width//100

window = pyglet.window.Window(width, height)

score_left = pyglet.text.Label(f"{score_p1}",  #text labels generated with chatgpt
                               font_name='Arial',
                               font_size=30,
                               color=YELLOW,
                               x=window.width - 20, y=window.height - 20,
                               anchor_x='right', anchor_y='top')
score_right = pyglet.text.Label(f"{score_p2}",
                                font_name='Arial',
                                font_size=30,
                                color=YELLOW,
                                x=20, y=window.height - 20,
                                anchor_x='left', anchor_y='top')
instructions = pyglet.text.Label('Welcome to PONG press 1(slow), 2(medium) or 3(fast) to start',
                                 font_name='Arial',
                                 font_size=10,
                                 color=YELLOW,
                                 x=window.width // 2, y=window.height // 2,
                                 anchor_x='center', anchor_y='top')
middle_line = pyglet.shapes.Rectangle(width//2, 0, width//100, height, YELLOW)

def order_points(pts): #taken from image extractor
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    print(rect)
    return rect


def convert_frame(coordinates, frame):  #taken from image extractor
    pts_src = np.array(coordinates)
    ordered_pts = pts_src
    # Define the destination points for the transformed image
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered_pts, pts_dst)
    frame = cv2.warpPerspective(frame, matrix, (int(width), int(height)))
    #transform into black and white filtering noise with gaussian blur and then applying binary threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)
    ret, threshold = cv2.threshold(gaussian, cutoff, 255, cv2.THRESH_BINARY_INV)
    return cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)


# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img, fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols,
                                   height=rows,
                                   fmt=fmt,
                                   data=raw_img,
                                   pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg

class Ball: #done mostly myself but structure help from chatgpt
    def __init__(self, x, y, radius, color, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vel_x = speed
        self.vel_y = speed
        self.speed = speed

    def update(self, frame):  #add frame
        global score_p1, score_p2
        self.x += self.vel_x
        self.y += self.vel_y
        # makes sure ball stays in play area
        if self.y <= 0 or self.y >= height:
            self.vel_y *= -1
        # checks if point scored, if so resets ball and updates points
        if self.x <= 0:
            self.x = width // 2
            self.y = height // 2
            self.vel_x *= -1
            score_p2 += 1
            score_left.text = f"{score_p2}"
        # same for other side
        if self.x >= width:
            self.x = width // 2
            self.y = height // 2
            self.vel_x *= -1
            score_p1 += 1
            score_right.text = f"{score_p1}"
        #checks ball location for collisions with player hand, whenever ball touches white changes direction
        flip_y = height - self.y - 1
        pixel_value = frame[flip_y, self.x]
        if (pixel_value == [255, 255, 255]).all():
            self.vel_x *= -1


    def draw(self):
        # creates the ball
        pyglet.shapes.Circle(self.x, self.y, self.radius, color=self.color).draw()


#starts game by initializing the ball
def startGame(speed):
    global ball, game_started
    game_started = True
    ball = Ball(width // 2, height // 2, 10, ORANGE, speed)


@window.event
def on_draw():
    isRunning = False
    window.clear()
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    marker_coordinates = [[0, 0], [0, 0], [0, 0], [0, 0]]
    # Check if marker is detected
    if ids is not None:
        #get coordinates of the markers, helped by chatpgt
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                corner_0 = corners[i][0][0]
                marker_coordinates[marker_id] = [corner_0[0], corner_0[1]]
                #marker_coordinates.append([corner_0[0], corner_0[1]])
        if len(ids) == 4:
            frame = convert_frame(marker_coordinates, frame)
            isRunning = True
    frame = cv2.flip(frame, 2)
    img = cv2glet(frame, 'BGR')
    img.blit(0, 0, 0)
    if not game_started:
        instructions.draw()
    score_left.draw()
    score_right.draw()
    # checks that game has been started and markers are visible to update ball
    if game_started and isRunning:
        middle_line.draw()
        ball.update(frame)
        ball.draw()


@window.event
def on_key_press(symbol, modifiers):
    # q to quit 1, 2 or 3 for difficulty/ball speed and to start round
    if symbol == pyglet.window.key.Q:
        os._exit(0)
    if symbol == pyglet.window.key._1:
        startGame(SLOW)
    if symbol == pyglet.window.key._2:
        startGame(MEDIUM)
    if symbol == pyglet.window.key._3:
        startGame(FAST)


pyglet.app.run()
