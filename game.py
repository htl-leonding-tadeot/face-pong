import cv2
import numpy as np
import random
import math
import time

width = 1280
height = 720

# Make sure 'haarcascade_frontalface_default.xml' is in the same directory
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
    """Reset ball to center with random direction within 15 degrees"""
    ball.x = width/2
    ball.y = height/2

    # Random angle between -15 and 15 degrees
    angle = random.uniform(-15, 15)
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

# --- Game & Paddle Constants ---
PADDLE_WIDTH = 30
PADDLE_HEIGHT = 100
BALL_RADIUS = 9
PADDLE_INFLUENCE_FACTOR = 0.5 # How much paddle speed affects ball dy

# --- NEW: Angle Clamping Constants ---
MIN_ANGLE_DEG = 15
MAX_ANGLE_DEG = 60
# Pre-calculate tangents for efficiency
MIN_ANGLE_TAN = math.tan(math.radians(MIN_ANGLE_DEG))
MAX_ANGLE_TAN = math.tan(math.radians(MAX_ANGLE_DEG))


# --- AFK Bot Constants ---
AFK_TIMEOUT = 10.0
BOT_SPEED = 12
BOT_DEV_RANGE = 75
BOT_OVERSHOOT_FACTOR = 1.1
BOT_UPDATE_INTERVAL = 3

# --- Persistent Paddle Vertical Positions ---
leftPaddleY = height // 2 - PADDLE_HEIGHT // 2
rightPaddleY = height // 2 - PADDLE_HEIGHT // 2

# --- NEW: Previous Paddle Positions (for speed tracking) ---
prevLeftPaddleY = leftPaddleY
prevRightPaddleY = rightPaddleY

# --- Bot Target/Frame Tracking ---
left_bot_target_y = leftPaddleY + PADDLE_HEIGHT / 2
right_bot_target_y = rightPaddleY + PADDLE_HEIGHT / 2
frame_counter = 0

# --- AFK Time Tracking ---
current_time = time.time()
last_face_time_left = current_time
last_face_time_right = current_time

# --- Center line for assignment ---
center_x = width / 2


while True:
    current_time = time.time()
    frame_counter += 1

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

    # --- 2. Wall & Goal Collisions ---
    if ball.y > height - BALL_RADIUS:
        ball.y = height - BALL_RADIUS
        ball.dy *= -1
    if ball.y < BALL_RADIUS:
        ball.y = BALL_RADIUS
        ball.dy *= -1
    if ball.x > width - BALL_RADIUS:
        reset_ball(ball)
        leftScore = leftScore + 1
    if ball.x < BALL_RADIUS:
        reset_ball(ball)
        rightScore = rightScore + 1

    # --- 3. Location-based Paddle Assignment (Human Input) ---
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

    if best_left_y is not None:
        leftPaddleY = best_left_y
        last_face_time_left = current_time

    if best_right_y is not None:
        rightPaddleY = best_right_y
        last_face_time_right = current_time

    # --- 4. AFK Bot Logic ---

    # 4a. Update Bot Target
    if frame_counter % BOT_UPDATE_INTERVAL == 0:
        deviation = random.uniform(-BOT_DEV_RANGE, BOT_DEV_RANGE)
        base_target_y = ball.y + deviation

        if ball.dx < 0: # Left Bot tracks
            left_bot_target_y = base_target_y
        if ball.dx > 0: # Right Bot tracks
            right_bot_target_y = base_target_y

    # 4b. Left Paddle Bot Movement
    is_afk_left = False
    if current_time - last_face_time_left > AFK_TIMEOUT:
        is_afk_left = True
        paddle_center_y = leftPaddleY + PADDLE_HEIGHT / 2
        diff_y = (left_bot_target_y - paddle_center_y) * BOT_OVERSHOOT_FACTOR
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)
        leftPaddleY = int(np.clip(leftPaddleY + move_y, 0, height - PADDLE_HEIGHT))

    # 4c. Right Paddle Bot Movement
    is_afk_right = False
    if current_time - last_face_time_right > AFK_TIMEOUT:
        is_afk_right = True
        paddle_center_y = rightPaddleY + PADDLE_HEIGHT / 2
        diff_y = (right_bot_target_y - paddle_center_y) * BOT_OVERSHOOT_FACTOR
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)
        rightPaddleY = int(np.clip(rightPaddleY + move_y, 0, height - PADDLE_HEIGHT))

    # --- 4d. NEW: Calculate Paddle Speeds ---
    # (This is done *after* human/bot logic moves the paddles)
    leftPaddleSpeed = leftPaddleY - prevLeftPaddleY
    rightPaddleSpeed = rightPaddleY - prevRightPaddleY


    # --- 5. Paddle Drawing and Collision Check ---

    paddle_data = [
        (0, leftPaddleY, is_afk_left, leftPaddleSpeed),
        (1, rightPaddleY, is_afk_right, rightPaddleSpeed)
    ]

    for index, paddle_y, is_afk, paddle_speed in paddle_data:
        paddle_x = 100 + (index * paddleX)

        # Draw paddle
        color = (0, 0, 255) if not is_afk else (255, 0, 0) # Red for Human, Blue for Bot
        cv2.rectangle(img, (paddle_x, int(paddle_y)), (paddle_x + PADDLE_WIDTH, int(paddle_y + PADDLE_HEIGHT)), color, -1)

        # Collision Check:
        if paddle_y - BALL_RADIUS <= ball.y <= paddle_y + PADDLE_HEIGHT + BALL_RADIUS:

            # --- Left paddle (index == 0) ---
            if index == 0:
                if ball.dx < 0 and (paddle_x + PADDLE_WIDTH) >= (ball.x - BALL_RADIUS) >= paddle_x:
                    ball.dx *= -1
                    ball.x = paddle_x + PADDLE_WIDTH + BALL_RADIUS

                    # --- NEW: Apply Paddle Influence & Clamp Angle ---
                    ball.dy += paddle_speed * PADDLE_INFLUENCE_FACTOR

                    min_abs_dy = abs(ball.dx) * MIN_ANGLE_TAN
                    max_abs_dy = abs(ball.dx) * MAX_ANGLE_TAN

                    sign_dy = np.sign(ball.dy)
                    clamped_abs_dy = np.clip(abs(ball.dy), min_abs_dy, max_abs_dy)
                    ball.dy = clamped_abs_dy * sign_dy

            # --- Right paddle (index == 1) ---
            else:
                if ball.dx > 0 and paddle_x <= (ball.x + BALL_RADIUS) <= (paddle_x + PADDLE_WIDTH):
                    ball.dx *= -1
                    ball.x = paddle_x - BALL_RADIUS

                    # --- NEW: Apply Paddle Influence & Clamp Angle ---
                    ball.dy += paddle_speed * PADDLE_INFLUENCE_FACTOR

                    min_abs_dy = abs(ball.dx) * MIN_ANGLE_TAN
                    max_abs_dy = abs(ball.dx) * MAX_ANGLE_TAN

                    sign_dy = np.sign(ball.dy)
                    clamped_abs_dy = np.clip(abs(ball.dy), min_abs_dy, max_abs_dy)
                    ball.dy = clamped_abs_dy * sign_dy

    # --- 6. Drawing and Display ---
    cv2.circle(img, (int(ball.x), int(ball.y)), BALL_RADIUS, (0, 0, 255), -1)

    left_status = "Bot" if is_afk_left else "Human"
    right_status = "Bot" if is_afk_right else "Human"

    text1 = cv2.putText(img, f'L ({left_status}): {leftScore}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    text2 = cv2.putText(img, f'R ({right_status}): {rightScore}', (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, img)

    # --- 7. Exit Condition ---
    k = cv2.waitKey(30) & 0xff
    if k == 27: # ESC key
        break

    # --- 8. NEW: Update previous paddle positions for next frame ---
    prevLeftPaddleY = leftPaddleY
    prevRightPaddleY = rightPaddleY

cap.release()
cv2.destroyAllWindows()