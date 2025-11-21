import cv2
import numpy as np
import random
import math

width = 1280
height = 720

# Make sure 'haarcascade_frontalface_default.xml' is in the same directory,
# or provide the full path to the file.
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, width)
cap.set(4, height)

window_name = "image"

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


class Vec:
    # Used for both ball movement (x, y, dx, dy) and face coordinates (x, y, w, h)
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy


def reset_ball(ball, speed=15):
    """Reset ball to center with random direction within 45 degrees"""
    ball.x = width/2
    ball.y = height/2

    # Random angle between -45 and 45 degrees
    angle = random.uniform(-45, 45)
    angle_rad = math.radians(angle)

    # Random direction (left or right)
    direction = random.choice([-1, 1])

    # Calculate velocity components
    ball.dx = direction * speed * math.cos(angle_rad)
    ball.dy = speed * math.sin(angle_rad)


paddleX = width - 230 # Used to define the paddle's x-coordinate (100 + (index * paddleX))

ball = Vec(100, 100, 10, 10)
reset_ball(ball)

leftScore = 0
rightScore = 0

# Define constants for collision check
PADDLE_WIDTH = 30
PADDLE_HEIGHT = 100
BALL_RADIUS = 9 # Based on the drawing cv2.circle(..., 9, ...)

# --- NEW: Persistent paddle vertical positions ---
# Initialize both paddles to the vertical center of the screen
leftPaddleY = height // 2 - PADDLE_HEIGHT // 2
rightPaddleY = height // 2 - PADDLE_HEIGHT // 2


while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Face Detection ---
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

    # --- 1. Ball Movement ---
    ball.x += ball.dx
    ball.y += ball.dy

    # --- 2. Top/Bottom Wall Collisions ---
    if ball.y > height - BALL_RADIUS:
        ball.y = height - BALL_RADIUS
        ball.dy *= -1

    if ball.y < BALL_RADIUS:
        ball.y = BALL_RADIUS
        ball.dy *= -1

    # --- 3. Left/Right Goal Collisions (Scoring) ---
    if ball.x > width - BALL_RADIUS:
        # Ball went past the right side (Left Player scores)
        reset_ball(ball)
        leftScore = leftScore + 1

    if ball.x < BALL_RADIUS:
        # Ball went past the left side (Right Player scores)
        reset_ball(ball)
        rightScore = rightScore + 1

    # --- 4. Update Persistent Paddle Positions based on detected faces ---
    faceCords = []
    for (x, y, w, h) in faces:
        faceCords.append(Vec(x, y, w, h))

    # Sort faces by x-coordinate to consistently identify Left (index 0) and Right (index 1) players
    faceCords.sort(key=lambda c: c.x)

    for index, vec in enumerate(faceCords[0:2], start=0):
        # The face's y-coordinate (vec.y) is the new paddle position
        if index == 0:
            # Update the Left paddle's position
            leftPaddleY = vec.y
        else: # index == 1
            # Update the Right paddle's position
            rightPaddleY = vec.y

    # --- 5. Paddle Drawing and Collision Check (Using persistent Y values) ---

    # Paddle data structure: (index, paddle_y_position)
    paddle_data = [
        (0, leftPaddleY),
        (1, rightPaddleY)
    ]

    for index, paddle_y in paddle_data:
        # The X position is fixed for both paddles
        paddle_x = 100 + (index * paddleX)

        # Draw the paddle
        cv2.rectangle(img, (paddle_x, paddle_y), (paddle_x + PADDLE_WIDTH, paddle_y + PADDLE_HEIGHT), (0, 0, 255), -1)

        # Collision Check: Is the ball vertically aligned with the paddle?
        if paddle_y - BALL_RADIUS <= ball.y <= paddle_y + PADDLE_HEIGHT + BALL_RADIUS:

            # Left paddle (index == 0) - Positioned on the left side
            if index == 0:
                # Check 1: Ball must be moving towards the paddle (left, ball.dx < 0)
                # Check 2: Ball's right edge (ball.x - BALL_RADIUS) is hitting the paddle's right edge
                if ball.dx < 0 and (paddle_x + PADDLE_WIDTH) >= (ball.x - BALL_RADIUS) >= paddle_x:
                    ball.dx *= -1  # Reverse horizontal direction
                    # Reposition ball just outside the paddle's right edge
                    ball.x = paddle_x + PADDLE_WIDTH + BALL_RADIUS

            # Right paddle (index == 1) - Positioned on the right side
            else:
                # Check 1: Ball must be moving towards the paddle (right, ball.dx > 0)
                # Check 2: Ball's left edge (ball.x + BALL_RADIUS) is hitting the paddle's left edge
                if ball.dx > 0 and paddle_x <= (ball.x + BALL_RADIUS) <= (paddle_x + PADDLE_WIDTH):
                    ball.dx *= -1  # Reverse horizontal direction
                    # Reposition ball just outside the paddle's left edge
                    ball.x = paddle_x - BALL_RADIUS

    # --- 6. Drawing and Display ---
    cv2.circle(img, (int(ball.x), int(ball.y)), BALL_RADIUS, (0, 0, 255), -1)

    text1 = cv2.putText(img, 'Left Player Score: ' + str(leftScore), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    text2 = cv2.putText(img, 'Right Player Score: ' + str(rightScore), (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, img)

    # --- 7. Exit Condition ---
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()