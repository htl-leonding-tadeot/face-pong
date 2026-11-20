import cv2
import numpy as np
import random
import math
import time
import subprocess  # For running shell commands
import os          # For restarting the script
import sys         # For restarting the script
import threading   # Camera + face detection run off the render thread
import glob        # Enumerating /dev/video* for camera autodetect
import re

width = 1280
height = 720

# Fraction of the frame held out of play at the top and bottom. The paddle follows the face
# one-to-one, and a face at the very frame edge is both awkward to reach and the first thing
# the detector loses, so the court stops short of it. Faces in the strips are still tracked and
# just pin the paddle at the court limit, which is how you reach the extremes.
COURT_PADDING = 0.10
COURT_TOP = round(height * COURT_PADDING)
COURT_BOTTOM = height - COURT_TOP

# YuNet (OpenCV 4.5.4+). Input size is set per-frame in detect_loop(). Paths here are
# relative to the cwd, like the git calls in check_for_update(): both assume the repo root.
face_detector = cv2.FaceDetectorYN.create(
    './face_detection_yunet_2023mar.onnx', '', (320, 320),
    score_threshold=0.7, nms_threshold=0.3, top_k=50)

IS_LINUX = sys.platform.startswith("linux")

if IS_LINUX and "GDK_BACKEND" not in os.environ and os.environ.get("DISPLAY"):
    # OpenCV's GTK window has no frame sync on GDK's Wayland backend: waitKey() blocks ~21 ms, which
    # caps the loop at ~34 fps and makes the fullscreen surface flicker. Via XWayland it's ~3.5 ms.
    # Must be set before the first cv2 window call (that's where GTK initialises). macOS uses Cocoa.
    os.environ["GDK_BACKEND"] = "x11"

if IS_LINUX:
    # Letting detection fan out over all 16 cores costs ~40% more CPU for no extra fps, and the
    # OpenMP pool's spin-wait is what made the laptop hot. macOS keeps OpenCV's own default.
    cv2.setNumThreads(int(os.environ.get("CV_THREADS", 4)))


def open_camera(index):
    """Open one camera index at the best mode (Nikon Z50II: 1080p60 MJPEG); frames get resized to width x height."""
    # Linux needs the V4L2 backend: the GStreamer one OpenCV prefers fails to negotiate with UVC
    # ("Internal data stream error"). macOS keeps CAP_ANY -> AVFoundation, which worked as-is.
    c = cv2.VideoCapture(index, cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_ANY)
    if IS_LINUX:
        # V4L2 opens in YUYV, which has no 1080p60 mode on either camera here; MJPEG does
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    c.set(cv2.CAP_PROP_FPS, 60)
    if not c.isOpened():
        c.release()
        return None
    if IS_LINUX and not c.read()[0]:
        c.release()                 # Opens fine but yields nothing: a UVC metadata node, not a camera
        return None
    return c


def camera_candidates():
    """Camera indices to try, preferred first; device order differs per machine."""
    if "CAMERA" in os.environ:
        return [int(os.environ["CAMERA"])]
    if not IS_LINUX:
        return [0]                  # macOS: unchanged, AVFoundation lists the Nikon first when connected
    # Linux enumerates differently (Nikon = /dev/video2 here, and video1/3 are metadata nodes),
    # so pick by device name instead of a hardcoded index.
    want = os.environ.get("CAMERA_NAME", "NIKON").lower()

    def name(i):
        try:
            with open(f"/sys/class/video4linux/video{i}/name") as f:
                return f.read().strip().lower()
        except OSError:
            return ""

    idx = sorted(int(re.search(r"\d+$", p).group()) for p in glob.glob("/dev/video*"))
    return sorted(idx, key=lambda i: want not in name(i))  # name match first, else ascending


cap = next((c for c in map(open_camera, camera_candidates()) if c), None)
if cap is None:
    sys.exit("No usable camera found. Set CAMERA=<index> to pick one.")

window_name = "image"


# --- Helper function to check for git updates ---
def check_for_update():
    """
    Fetches from remote, compares local and remote hashes, and pulls if different.
    Returns True if an update was pulled, False otherwise.
    """
    try:
        print("Checking for updates...")
        # 1. Fetch updates from remote without applying them
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        # 2. Get the commit hash of the local HEAD
        local_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True
        ).stdout.strip()

        # 3. Get the commit hash of the remote branch (upstream)
        remote_hash = subprocess.run(
            ["git", "rev-parse", "@{u}"],
            check=True, capture_output=True, text=True
        ).stdout.strip()

        if local_hash != remote_hash:
            print(f"New update found (Local: {local_hash[:7]}, Remote: {remote_hash[:7]}). Pulling changes...")
            # 4. Pull the changes
            subprocess.run(["git", "pull"], check=True)
            return True # Update was pulled
        else:
            print("No update found. Already up-to-date.")
            return False # Already up-to-date

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print("Please ensure you are in a git repository and the upstream branch is set.")
        return False # Error, don't restart
    except FileNotFoundError:
        print("Git command not found. Please ensure git is installed and in your PATH.")
        return False # Error, don't restart

# --- Helper function to restart the script ---
def restart_script():
    """
    Restarts the current Python script, replacing the current process.
    """
    print("Restarting script with new code...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


class Vec:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy


# --- Timing ---
TICK_RATE = 60              # Physics ticks per second (fixed)
TICK = 1.0 / TICK_RATE
MAX_FPS = 60                # Render cap
FRAME_TIME = 1.0 / MAX_FPS
# Tuning constants below were written for the old ~30 fps per-frame loop.
# K converts "per old frame" to "per tick" so gameplay speed stays the same.
DESIGN_FPS = 30
K = DESIGN_FPS / TICK_RATE


def reset_ball(ball, speed=15 * K):
    """Reset ball to center with random direction within 15 degrees"""
    ball.x = width/2
    ball.y = (COURT_TOP + COURT_BOTTOM) / 2

    # Random angle between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    angle_rad = math.radians(angle)

    # Random direction (left or right)
    direction = random.choice([-1, 1])

    # Calculate velocity components
    ball.dx = direction * speed * math.cos(angle_rad)
    ball.dy = speed * math.sin(angle_rad)


# --- Camera + detection threads, both off the render thread ---
# Detection is still the most expensive step, so it runs on its own thread at its own rate
# instead of once per camera frame. Capture and render stay at 60 fps regardless.
DETECT_SCALE = 0.5          # Detect on a half-size frame (~3x cheaper); coords scale back up. 1.0 = full frame
DETECT_MAX_HZ = 60          # Upper bound on detections per second (YuNet is ~4.7 ms a frame)

latest_img = np.zeros((height, width, 3), np.uint8)
latest_faces = ()
frame_seq = 0               # Increments per camera frame
face_seq = 0                # Increments per completed detection
camera_ok = True
stop_event = threading.Event()
cam_lock = threading.Lock()
frame_ready = threading.Condition(cam_lock)


def camera_loop():
    global latest_img, frame_seq, camera_ok
    while not stop_event.is_set():
        ret, frame = cap.read()  # Blocks until next camera frame (releases GIL)
        if not ret:
            camera_ok = False
            with frame_ready:
                frame_ready.notify_all()    # Don't leave the detector blocked on a dead camera
            return
        # Camera may ignore the requested resolution; game logic assumes width x height.
        # Cover, not contain: centre-crop to 16:9 first, so a camera that isn't 16:9 loses the
        # edges instead of being stretched. Both cameras here are 16:9, where this is a no-op.
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        cw, ch = min(fw, round(fh * width / height)), min(fh, round(fw * height / width))
        y0, x0 = (fh - ch) // 2, (fw - cw) // 2
        frame = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (width, height))
        with frame_ready:
            latest_img = frame              # Fresh array each frame, so readers can hold it unlocked
            frame_seq += 1
            frame_ready.notify_all()


def detect_loop():
    global latest_faces, face_seq
    seen = 0
    while not stop_event.is_set():
        started = time.perf_counter()
        with frame_ready:
            # Wait for a frame newer than the last one detected on; older ones are dropped
            frame_ready.wait_for(lambda: frame_seq != seen or stop_event.is_set() or not camera_ok, 0.5)
            if not camera_ok:
                return
            frame, seen = latest_img, frame_seq
        # YuNet takes BGR, and returns Nx15: box x,y,w,h, then five (x,y) landmarks in the
        # order right eye, left eye, nose tip, right mouth corner, left mouth corner, then score
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE) if DETECT_SCALE != 1.0 else frame
        face_detector.setInputSize((small.shape[1], small.shape[0]))
        _, dets = face_detector.detect(small)
        # (center_x, eye_mid_y, chin_y) per face. The eye midpoint is hair-independent, unlike
        # the box top edge, which moves with a fringe; chin_y is where the marker hangs.
        # dets is None (not an empty array) when nothing is found
        faces = [((d[0] + d[2] / 2) / DETECT_SCALE, (d[5] + d[7]) / 2 / DETECT_SCALE,
                  (d[1] + d[3]) / DETECT_SCALE)
                 for d in (dets if dets is not None else ())]
        with cam_lock:
            latest_faces = faces
            face_seq += 1
        # Detection is the one CPU hog; leave the cores alone if it finished early
        slack = 1.0 / DETECT_MAX_HZ - (time.perf_counter() - started)
        if slack > 0:
            time.sleep(slack)


cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()
detect_thread = threading.Thread(target=detect_loop, daemon=True)
detect_thread.start()

# Wait for the first real frame before opening the window, so it never shows uninitialised content
with frame_ready:
    frame_ready.wait_for(lambda: frame_seq > 0 or not camera_ok, 5.0)

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


paddleX = width - 230

ball = Vec(100, 100, 10, 10)
reset_ball(ball)



COUNTDOWN_SEC = 0.5         # Per number, so 3-2-1 runs 1.5 s
countdown_until = 0.0       # Ball is held at centre until this time


def reset_game(reason, countdown=False):
    """Zero every score and recentre the ball. Only the deliberate resets get a 3-2-1."""
    global countdown_until
    for t in tracks.values():
        t['score'] = 0
    bot_scores[0] = bot_scores[1] = 0
    reset_ball(ball)
    if countdown:
        countdown_until = time.time() + 3 * COUNTDOWN_SEC
    print(f"Game reset: {reason}")

# --- Marker colours ---
PLAYER_COLOR = (0, 165, 255)    # Orange, matching a human's paddle
BYSTANDER_COLOR = (150, 150, 150)   # Grey: tracked and numbered, but not holding a slot

# --- Game & Paddle Constants ---
PADDLE_WIDTH = 30
PADDLE_HEIGHT = 100
BALL_RADIUS = 9
PADDLE_INFLUENCE_FACTOR = 0.5
HUMAN_SMOOTHING = 0.5       # Per tick: fraction of gap to face position closed (camera is only 30 fps)

# --- Angle Clamping Constants ---
MIN_ANGLE_DEG = 15
MAX_ANGLE_DEG = 60
MIN_ANGLE_TAN = math.tan(math.radians(MIN_ANGLE_DEG))
MAX_ANGLE_TAN = math.tan(math.radians(MAX_ANGLE_DEG))


# --- Bot Constants (a bot plays any side whose slot is empty; see TRACK_TTL) ---
BOT_SPEED = 12 * K
BOT_DEV_RANGE = 75
BOT_OVERSHOOT_FACTOR = 1.1
BOT_UPDATE_INTERVAL = max(1, round(3 / K))       # In ticks (same reaction time as before)
BOT_SMOOTHING_FACTOR = 1 - (1 - 0.2) ** K        # 20% per old frame, rescaled per tick

# --- Persistent Paddle Vertical Positions ---
leftPaddleY = height // 2 - PADDLE_HEIGHT // 2
rightPaddleY = height // 2 - PADDLE_HEIGHT // 2
leftFaceY = leftPaddleY     # Latest detected face position (human target)
rightFaceY = rightPaddleY

# --- Previous Paddle Positions (for speed tracking) ---
prevLeftPaddleY = leftPaddleY
prevRightPaddleY = rightPaddleY

# --- Bot Target/Tick Tracking ---
raw_left_bot_target_y = leftPaddleY + PADDLE_HEIGHT / 2
raw_right_bot_target_y = rightPaddleY + PADDLE_HEIGHT / 2
smooth_left_bot_target_y = leftPaddleY + PADDLE_HEIGHT / 2
smooth_right_bot_target_y = rightPaddleY + PADDLE_HEIGHT / 2
tick_counter = 0

current_time = time.time()

# --- People tracking -------------------------------------------------------
# The detector has no notion of identity: every frame it just reports boxes. Faces are matched
# to the previous frame's by position, which is enough here because people walk, they do not
# teleport. A track outlives TRACK_TTL seconds of missed detections, so looking away or being
# briefly hidden behind someone costs nothing; walking off costs the slot and the score.
TRACK_TTL = 3.0             # Seconds a track (and its slot) survives with no detection
TRACK_MATCH_DIST = 180      # Max px a face may move between detections and still be one person
TRACK_MATCH_DRIFT = 400     # Extra px of search per second unseen: people keep walking while
                            # the detector has lost them, so the window has to open up with time

tracks = {}                 # track id -> player state; see new_track()
next_track_id = 0
next_player_number = 1      # Reset to 1 whenever the field empties, so numbers get reused
slots = [None, None]        # Track id playing each side, None when nobody holds it
bot_scores = [0, 0]         # What the bot has earned on a side since the slot came free
is_afk_left = False         # "AFK" now just means this side's slot is empty
is_afk_right = False

# --- Auto-Update Tracking ---
UPDATE_CHECK_INTERVAL = 60 # 1 minute (in seconds)
last_update_check_time = current_time
update_check_pending = False
was_idle_last_frame = False # Check for transition to idle


# --- Center line for assignment ---
center_x = width / 2


def side_of(x):
    """Which half of the frame a face is standing in."""
    return 0 if x < center_x else 1


def update_tracks(faces, now):
    """Match this frame's faces to the people already being followed."""
    global next_track_id, next_player_number

    free = dict(tracks)                     # Tracks not yet claimed by a face this frame
    for (cx, eye_y, chin_y) in faces:
        best, best_score = None, 1.0
        for tid, t in free.items():
            limit = TRACK_MATCH_DIST + TRACK_MATCH_DRIFT * (now - t['last_seen'])
            # Scored as a fraction of each track's own window, so a long-unseen track competes
            # fairly with one seen last frame instead of always losing on raw distance
            score = math.hypot(cx - t['x'], eye_y - t['eye_y']) / limit
            if score < best_score:
                best, best_score = tid, score
        if best is None:                    # Nobody nearby was here before, so this is a newcomer
            best = next_track_id
            next_track_id += 1
            tracks[best] = {'number': next_player_number, 'score': 0}
            next_player_number += 1
        else:
            del free[best]
        tracks[best].update(x=cx, eye_y=eye_y, chin_y=chin_y, last_seen=now, visible=True)

    for tid, t in free.items():             # Seen before, not in this frame
        t['visible'] = False
    for tid in [tid for tid, t in tracks.items() if now - t['last_seen'] > TRACK_TTL]:
        del tracks[tid]                     # Gone long enough to count as gone


def update_slots():
    """Hand out and take back the one playing slot each side has."""
    global is_afk_left, is_afk_right

    for side in (0, 1):
        tid = slots[side]
        # A slot is only ever given up by its own holder: by leaving, or by crossing over.
        # Other people crowding onto the side cannot take it off them.
        if tid is not None and (tid not in tracks or side_of(tracks[tid]['x']) != side):
            slots[side] = None
            bot_scores[side] = 0            # The bot plays for itself, from zero

    for side in (0, 1):
        if slots[side] is None:
            here = [tid for tid, t in tracks.items() if side_of(t['x']) == side]
            if len(here) == 1:              # Exactly one person on that side, so no ambiguity
                slots[side] = here[0]

    is_afk_left = slots[0] is None
    is_afk_right = slots[1] is None


def side_label(side):
    """What to call whoever is playing a side, matching the marker under their face."""
    tid = slots[side]
    return "Bot" if tid is None else f"Player {tracks[tid]['number']}"


def side_score(side):
    tid = slots[side]
    return bot_scores[side] if tid is None else tracks[tid]['score']


def add_point(side):
    tid = slots[side]
    if tid is None:
        bot_scores[side] += 1
    else:
        tracks[tid]['score'] += 1           # Points belong to the person, and travel with them


def bounce(paddle_speed):
    """Apply paddle influence & clamp angle after a paddle hit"""
    ball.dy += paddle_speed * PADDLE_INFLUENCE_FACTOR
    min_abs_dy = abs(ball.dx) * MIN_ANGLE_TAN
    max_abs_dy = abs(ball.dx) * MAX_ANGLE_TAN
    sign_dy = np.sign(ball.dy)
    clamped_abs_dy = np.clip(abs(ball.dy), min_abs_dy, max_abs_dy)
    ball.dy = clamped_abs_dy * sign_dy


def tick(faces, new_faces):
    """Advance the game by exactly one fixed timestep"""
    global leftPaddleY, rightPaddleY, leftFaceY, rightFaceY
    global prevLeftPaddleY, prevRightPaddleY, next_player_number
    global raw_left_bot_target_y, raw_right_bot_target_y
    global smooth_left_bot_target_y, smooth_right_bot_target_y
    global tick_counter, is_afk_left, is_afk_right
    tick_counter += 1

    # --- 1. Ball Movement (held still through the countdown; paddles keep tracking) ---
    counting_down = current_time < countdown_until
    if not counting_down:
        ball.x += ball.dx
        ball.y += ball.dy

        # --- 2. Wall & Goal Collisions ---
        if ball.y > COURT_BOTTOM - BALL_RADIUS:
            ball.y = COURT_BOTTOM - BALL_RADIUS
            ball.dy *= -1
        if ball.y < COURT_TOP + BALL_RADIUS:
            ball.y = COURT_TOP + BALL_RADIUS
            ball.dy *= -1
        if ball.x > width - BALL_RADIUS:
            reset_ball(ball)
            add_point(0)
        if ball.x < BALL_RADIUS:
            reset_ball(ball)
            add_point(1)

    # --- 3. Follow the people in frame (only a fresh camera frame carries new positions) ---
    if new_faces:
        was_occupied = bool(tracks)
        update_tracks(faces, current_time)
        update_slots()
        if was_occupied and not tracks:
            # Field empty. Numbers start over at 1 and nothing is carried into the next game.
            next_player_number = 1
            bot_scores[0] = bot_scores[1] = 0
            reset_game("field empty")

    # Paddle target is the slot holder's eyes. A face out in the dimmed strips still counts;
    # it clamps to the court limit, so the extremes are reachable without going to the frame edge
    for side, tid in enumerate(slots):
        if tid is None:
            continue
        y = tracks[tid]['eye_y'] - PADDLE_HEIGHT / 2   # Centre the paddle on the eyes
        y = min(max(y, COURT_TOP), COURT_BOTTOM - PADDLE_HEIGHT)
        if side == 0:
            leftFaceY = y
        else:
            rightFaceY = y

    # Humans glide toward their face position so 30 fps camera input looks smooth at 60 ticks
    if not is_afk_left:
        leftPaddleY += (leftFaceY - leftPaddleY) * HUMAN_SMOOTHING
    if not is_afk_right:
        rightPaddleY += (rightFaceY - rightPaddleY) * HUMAN_SMOOTHING

    # --- 4. AFK Bot Logic ---

    # 4a. Update Bot's RAW Target (Reaction Time)
    if tick_counter % BOT_UPDATE_INTERVAL == 0:
        deviation = random.uniform(-BOT_DEV_RANGE, BOT_DEV_RANGE)
        base_target_y = ball.y + deviation

        if ball.dx < 0: # Left Bot tracks
            raw_left_bot_target_y = base_target_y
        if ball.dx > 0: # Right Bot tracks
            raw_right_bot_target_y = base_target_y

    # 4b. Update Bot's SMOOTH Target (Movement Fluidity)
    smooth_left_bot_target_y = (smooth_left_bot_target_y * (1 - BOT_SMOOTHING_FACTOR)) + (raw_left_bot_target_y * BOT_SMOOTHING_FACTOR)
    smooth_right_bot_target_y = (smooth_right_bot_target_y * (1 - BOT_SMOOTHING_FACTOR)) + (raw_right_bot_target_y * BOT_SMOOTHING_FACTOR)

    # 4c. Left Paddle Bot Movement
    if is_afk_left:
        paddle_center_y = leftPaddleY + PADDLE_HEIGHT / 2
        diff_y = (smooth_left_bot_target_y - paddle_center_y) * BOT_OVERSHOOT_FACTOR
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)
        leftPaddleY = float(np.clip(leftPaddleY + move_y, COURT_TOP, COURT_BOTTOM - PADDLE_HEIGHT))

    # 4d. Right Paddle Bot Movement
    if is_afk_right:
        paddle_center_y = rightPaddleY + PADDLE_HEIGHT / 2
        diff_y = (smooth_right_bot_target_y - paddle_center_y) * BOT_OVERSHOOT_FACTOR
        move_y = np.clip(diff_y, -BOT_SPEED, BOT_SPEED)
        rightPaddleY = float(np.clip(rightPaddleY + move_y, COURT_TOP, COURT_BOTTOM - PADDLE_HEIGHT))

    # 4e. Calculate Paddle Speeds (per tick, same unit as ball.dy)
    leftPaddleSpeed = leftPaddleY - prevLeftPaddleY
    rightPaddleSpeed = rightPaddleY - prevRightPaddleY
    prevLeftPaddleY = leftPaddleY
    prevRightPaddleY = rightPaddleY

    # --- 5. Paddle Collision Check ---
    for index, paddle_y, paddle_speed in ((0, leftPaddleY, leftPaddleSpeed), (1, rightPaddleY, rightPaddleSpeed)):
        paddle_x = 100 + (index * paddleX)
        if paddle_y - BALL_RADIUS <= ball.y <= paddle_y + PADDLE_HEIGHT + BALL_RADIUS:
            if index == 0:
                if ball.dx < 0 and (paddle_x + PADDLE_WIDTH) >= (ball.x - BALL_RADIUS) >= paddle_x:
                    ball.dx *= -1
                    ball.x = paddle_x + PADDLE_WIDTH + BALL_RADIUS
                    bounce(paddle_speed)
            else:
                if ball.dx > 0 and paddle_x <= (ball.x + BALL_RADIUS) <= (paddle_x + PADDLE_WIDTH):
                    ball.dx *= -1
                    ball.x = paddle_x - BALL_RADIUS
                    bounce(paddle_speed)


def render(img):
    # Out-of-play strips: the camera stays visible there (faces are still tracked) but dimmed,
    # with a line on each court edge so the play area reads from across a room
    img[:COURT_TOP] = img[:COURT_TOP] // 3
    img[COURT_BOTTOM:] = img[COURT_BOTTOM:] // 3
    cv2.line(img, (0, COURT_TOP), (width, COURT_TOP), (255, 255, 255), 2)
    cv2.line(img, (0, COURT_BOTTOM), (width, COURT_BOTTOM), (255, 255, 255), 2)

    for index, paddle_y, is_afk in ((0, leftPaddleY, is_afk_left), (1, rightPaddleY, is_afk_right)):
        paddle_x = 100 + (index * paddleX)
        # Orange for Human, Blue for Bot
        color = PLAYER_COLOR if not is_afk else (255, 0, 0)
        cv2.rectangle(img, (paddle_x, int(paddle_y)), (paddle_x + PADDLE_WIDTH, int(paddle_y + PADDLE_HEIGHT)), color, -1)

    cv2.circle(img, (int(ball.x), int(ball.y)), BALL_RADIUS, (0, 0, 255), -1)

    # Everyone in frame gets a marker and their number. Whoever holds a slot is in the player
    # colour, bystanders are grey, so it reads at a glance who the paddles belong to.
    playing = [tid for tid in slots if tid is not None]
    for tid, t in tracks.items():
        if not t['visible']:
            continue                        # Mid-dropout: the slot is held, but do not draw a
                                            # marker on a face that is not currently there
        color = PLAYER_COLOR if tid in playing else BYSTANDER_COLOR
        name = f"Player {t['number']}"
        mx, my = int(t['x']), int(t['chin_y']) + 10
        cv2.fillPoly(img, [np.array([(mx, my), (mx - 12, my + 16), (mx + 12, my + 16)])], color)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        org = (mx - tw // 2, min(my + 30 + th, height - 8))
        cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Scores. Sides are obvious from the layout, so the only label is who is playing:
    # the same name as under their face, or Bot when the slot is empty
    score_y = COURT_TOP + 40        # Inside the court; the strips are dimmed and this text is black
    right_text = f'{side_label(1)}: {side_score(1)}'
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(img, f'{side_label(0)}: {side_score(0)}', (50, score_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, right_text, (width - 50 - rw, score_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    # 3-2-1 over the middle of the court while the ball is held
    remaining = countdown_until - time.time()
    if remaining > 0:
        n = str(math.ceil(remaining / COUNTDOWN_SEC))
        (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, 6, 12)
        org = ((width - tw) // 2, (COURT_TOP + COURT_BOTTOM + th) // 2)
        cv2.putText(img, n, org, cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20, cv2.LINE_AA)
        cv2.putText(img, n, org, cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 12, cv2.LINE_AA)

    # Orange frame on the outermost pixels: marks where the camera image ends and the screen's
    # own letterboxing begins (highgui defaults to WINDOW_KEEPRATIO, so a 16:9 image on a 3:2
    # or 16:10 panel gets black bars that are otherwise indistinguishable from the image)
    cv2.rectangle(img, (1, 1), (width - 2, height - 2), (0, 165, 255), 2)

    cv2.imshow(window_name, img)


reset_game("startup", countdown=True)

seen_face_seq = 0
accumulator = 0.0
last_time = time.perf_counter()
next_frame = last_time

# FPS_DEBUG=1 prints render/camera/detect rates once a second
fps_debug = os.environ.get("FPS_DEBUG") == "1"
fps_mark, fps_frames, fps_cam, fps_det = last_time, 0, frame_seq, face_seq

while camera_ok:
    now = time.perf_counter()
    # ponytail: clamp catch-up so a stall (window drag, git fetch) doesn't spiral; game just slows briefly
    accumulator += min(now - last_time, 0.25)
    last_time = now
    current_time = time.time()

    with cam_lock:
        img, faces, seq = latest_img, latest_faces, face_seq

    # --- Fixed-timestep physics ---
    while accumulator >= TICK:
        tick(faces, seq != seen_face_seq)
        seen_face_seq = seq
        accumulator -= TICK

    # --- Render (copy: camera frame is shared and may be redrawn next frame) ---
    render(img.copy())

    if fps_debug:
        fps_frames += 1
        elapsed = now - fps_mark
        if elapsed >= 1.0:
            print("render %.1f fps | camera %.1f fps | detect %.1f fps"
                  % (fps_frames / elapsed, (frame_seq - fps_cam) / elapsed,
                     (face_seq - fps_det) / elapsed), flush=True)
            fps_mark, fps_frames, fps_cam, fps_det = now, 0, frame_seq, face_seq

    # --- Auto-Update Logic ---
    if current_time - last_update_check_time > UPDATE_CHECK_INTERVAL:
        print("Update interval passed. Flagging for update check.")
        update_check_pending = True
        last_update_check_time = current_time # Reset the update timer

    is_idle = is_afk_left and is_afk_right
    just_became_idle = is_idle and not was_idle_last_frame

    if update_check_pending and is_idle:
        if just_became_idle:
            print("Transitioned to dual bot mode with update pending. Checking now...")
        else:
            print("Idle mode and update timer elapsed. Checking now...")

        if check_for_update():
            restart_script() # This will stop the script

        update_check_pending = False

    was_idle_last_frame = is_idle

    # --- Exit Condition (waitKey(1) also pumps the window's event loop) ---
    k = cv2.waitKey(1) & 0xff
    if k == 27: # ESC key
        break
    if k in (ord('r'), ord('R')):   # Manual score reset
        reset_game("'r' pressed", countdown=True)

    # --- Frame cap: sleep only the time left until the next frame deadline ---
    next_frame += FRAME_TIME
    remaining = next_frame - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    else:
        next_frame = time.perf_counter()  # Running late: don't try to catch up on frames

stop_event.set()
cam_thread.join(timeout=1)
detect_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
