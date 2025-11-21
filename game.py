import cv2
import numpy as np
import random
import math
import time # Import time for AFK tracking

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


paddleX = width - 230

ball = Vec(100, 100, 10, 10)
reset_ball(ball)

leftScore = 0
rightScore = 0

# Define constants for collision check
PADDLE_WIDTH = 30
PADDLE_HEIGHT = 100
BALL_RADIUS = 9

# --- AFK Bot Constants ---
AFK_TIMEOUT = 10.0 # Time in seconds before bot takes over
BOT_SPEED = 15 # Speed at which the bot paddle moves

# --- Persistent Paddle Vertical Positions ---
leftPaddleY = height // 2 - PADDLE_HEIGHT // 2
rightPaddleY = height // 2 - PADDLE_HEIGHT // 2

# --- AFK Time Tracking ---
# Initialize the last time a face was seen to the current time
current_time = time.time()
last_face_time_left = current_time
last_face_time_right = current_time

# --- Center line for assignment ---
center_x = width / 2


while True:
    current_time = time.time() # Get current time at the start of the loop

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
        reset_ball(ball)
        leftScore = leftScore + 1

    if ball.x < BALL_RADIUS:
        reset_ball(ball)
        rightScore = rightScore + 1

    # --- 4. Location-based Paddle Assignment and AFK Time Update ---
    best_left_y = None
    best_right_y = None

    for (x, y, w, h) in faces:
        face_center_x = x + w / 2

        if face_center_x < center_x:
            if best_left_y is None or y < best_left_y:
                best_left_y = y
        else:
            if best_right_y is None or y < best_right_y:
                best_right_y = y

    # Update persistent paddle positions and reset AFK timers
    if best_left_y is not None:
        leftPaddleY = best_left_y
        last_face_time_left = current_time # Reset timer

    if best_right_y is not None:
        rightPaddleY = best_right_y
        last_face_time_right = current_time # Reset timer

    # --- 5. AFK Bot Logic ---

    # Target Y is the center of the ball
    target_y = int(ball.y)

    # Left Paddle Bot
    if current_time - last_face_time_left > AFK_TIMEOUT:
        is_afk_left = True
        # Calculate the direction and magnitude to move the paddle
        paddle_center_y = leftPaddleY + PADDLE_HEIGHT / 2
        diff_y = target_y - paddle_center_y

        # Determine movement amount (clamped by BOT_SPEED)
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)

        # Apply movement
        leftPaddleY += move_y

        # Clamp paddle position within screen bounds
        leftPaddleY = np.clip(leftPaddleY, 0, height - PADDLE_HEIGHT)

    else:
        is_afk_left = False

    # Right Paddle Bot
    if current_time - last_face_time_right > AFK_TIMEOUT:
        is_afk_right = True
        # Calculate the direction and magnitude to move the paddle
        paddle_center_y = rightPaddleY + PADDLE_HEIGHT / 2
        diff_y = target_y - paddle_center_y

        # Determine movement amount (clamped by BOT_SPEED)
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)

        # Apply movement
        rightPaddleY += move_y

        # Clamp paddle position within screen bounds
        rightPaddleY = np.clip(rightPaddleY, 0, height - PADDLE_HEIGHT)

    else:
        is_afk_right = False


    # --- 6. Paddle Drawing and Collision Check ---

    paddle_data = [
        (0, leftPaddleY, is_afk_left),
        (1, rightPaddleY, is_afk_right)
    ]

    for index, paddle_y, is_afk in paddle_data:
        paddle_x = 100 + (index * paddleX)

        # Use a different color for the bot paddle
        color = (0, 0, 255) if not is_afk else (255, 0, 0) # Red for Human, Blue for Bot

        # Draw the paddle
        cv2.rectangle(img, (paddle_x, int(paddle_y)), (paddle_x + PADDLE_WIDTH, int(paddle_y + PADDLE_HEIGHT)), color, -1)

        # Collision Check:
        if paddle_y - BALL_RADIUS <= ball.y <= paddle_y + PADDLE_HEIGHT + BALL_RADIUS:

            # Left paddle (index == 0)
            if index == 0:
                if ball.dx < 0 and (paddle_x + PADDLE_WIDTH) >= (ball.x - BALL_RADIUS) >= paddle_x:
                    ball.dx *= -1
                    ball.x = paddle_x + PADDLE_WIDTH + BALL_RADIUS

            # Right paddle (index == 1)
            else:
                if ball.dx > 0 and paddle_x <= (ball.x + BALL_RADIUS) <= (paddle_x + PADDLE_WIDTH):
                    ball.dx *= -1
                    ball.x = paddle_x - BALL_RADIUS

    # --- 7. Drawing and Display ---
    cv2.circle(img, (int(ball.x), int(ball.y)), BALL_RADIUS, (0, 0, 255), -1)

    # Display Score and AFK status
    left_status = "Bot" if is_afk_left else "Human"
    right_status = "Bot" if is_afk_right else "Human"

    text1 = cv2.putText(img, f'L ({left_status}): {leftScore}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    text2 = cv2.putText(img, f'R ({right_status}): {rightScore}', (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, img)

    # --- 8. Exit Condition ---
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()