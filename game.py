import os

# OpenCV's GTK window can be much slower on GTK's Wayland backend than through
# XWayland (42ms vs 6ms per fullscreen frame on an Apple M-series laptop), so
# prefer X11 when an X display is reachable. Must happen before the first window.
if os.environ.get('WAYLAND_DISPLAY') and os.environ.get('DISPLAY'):
    os.environ.setdefault('GDK_BACKEND', 'x11')

import cv2
import math
import numpy as np
import threading
import random
import time
import sys

# --- CONFIGURATION ---

# IMPORTANT: Make sure this path is correct!
CASCADE_PATH = r'./haarcascade_frontalface_default.xml'

# Game settings
BALL_RADIUS = 12        # at 720p; scaled for other resolutions
BALL_SPEED = 0.35       # screen heights per second
MAX_BALLS = 3
PADDLE_WIDTH = 120
PADDLE_HEIGHT = 20
PADDLE_Y_OFFSET = 50 # Distance from bottom
BLOCK_ROWS = 6
BLOCK_COLS = 10
BLOCK_HEIGHT = 30
POINTS_PER_BLOCK = 10

# Fraction of the frame held out of play on each side. The paddle follows the
# face, and a face near the frame edge is both hard to reach and hard to
# detect, so the walls sit where a player can comfortably get to them.
PLAYFIELD_MARGIN = 0.15
MIN_BOUNCE_ANGLE = 20   # degrees off horizontal, so rallies never flatten out

# Auto-Restart Configuration
INACTIVITY_TIMEOUT_SEC = 60.0
RESTART_ON_WIN = True

# Latency Configuration
DETECT_SCALE = 0.4      # face detection runs on a frame this much smaller

# Player Tracking Configuration (Kept from previous code)
TRACKING_SMOOTHING_FACTOR = 0.4
MAX_DISTANCE_SQUARED = 90000
DECAY_TIMEOUT = 1.5
MIN_CONSECUTIVE_DETECTIONS = 2

# --- GAME STATE ---
balls = []
blocks = []
players = {}
pending_faces = {}
next_player_id = 1
next_pending_id = 1
game_score = 0

# --- HELPER FUNCTIONS ---

def restart_game(screen_width, screen_height):
    """Resets the game back to its starting state.

    This used to re-exec the interpreter, which left the camera device open in
    the dying process and so failed to reopen it.
    """
    global balls, players, pending_faces, game_score
    print("--- GAME RESTARTING ---")
    balls = []
    players = {}
    pending_faces = {}
    game_score = 0
    create_blocks(screen_width, screen_height)
    spawn_ball(screen_width, screen_height)

def playfield(screen_width):
    """Left and right walls of the play area."""
    margin = round(screen_width * PLAYFIELD_MARGIN)
    return margin, screen_width - margin


def create_blocks(screen_width, screen_height):
    """Generates the grid of blocks, filling the play area exactly."""
    global blocks
    blocks = []

    field_left, field_right = playfield(screen_width)
    block_width = (field_right - field_left) / BLOCK_COLS

    colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 127, 255), (139, 0, 255)]

    for row in range(BLOCK_ROWS):
        color = colors[row % len(colors)]
        for col in range(BLOCK_COLS):
            b_x = field_left + col * block_width
            b_y = row * BLOCK_HEIGHT + 60 # 60px buffer from top

            # Add a small gap between blocks
            blocks.append({
                'rect': [b_x + 2, b_y + 2, block_width - 4, BLOCK_HEIGHT - 4],
                'color': color,
                'active': True
            })

def spawn_ball(screen_width, screen_height):
    """Creates a new ball moving downwards."""
    global balls
    if len(balls) < MAX_BALLS:
        # Spawn in the middle
        start_x = screen_width // 2
        start_y = screen_height // 2

        # Randomize angle downwards
        angle = random.uniform(math.pi/4, 3*math.pi/4)
        speed = BALL_SPEED * screen_height
        new_ball = {
            'pos': [start_x, start_y],
            'vel': [speed * math.cos(angle), speed * math.sin(angle)],
            'radius': BALL_RADIUS,
            'color': (255, 255, 255)
        }
        balls.append(new_ball)

def scale_to_frame(screen_height):
    """Scales the pixel sizes, which are tuned for 720p, to the camera's frame.

    Whatever resolution the camera gives us, the game should look the same.
    """
    global BALL_RADIUS, PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_Y_OFFSET
    global BLOCK_HEIGHT, MAX_DISTANCE_SQUARED
    factor = screen_height / 720
    BALL_RADIUS = max(4, round(BALL_RADIUS * factor))
    PADDLE_WIDTH = round(PADDLE_WIDTH * factor)
    PADDLE_HEIGHT = max(4, round(PADDLE_HEIGHT * factor))
    PADDLE_Y_OFFSET = round(PADDLE_Y_OFFSET * factor)
    BLOCK_HEIGHT = max(8, round(BLOCK_HEIGHT * factor))
    MAX_DISTANCE_SQUARED = round(MAX_DISTANCE_SQUARED * factor * factor)


class Camera:
    """Camera that always hands back the newest frame.

    The game loop is slower than the camera's 30 fps. Reading in the loop means
    the driver's buffers stay full and read() returns a frame that is several
    frames old; worse, the loop then blocks waiting its turn. A thread that
    does nothing but read keeps those buffers drained, so the loop gets the
    live frame without waiting.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.ok, self.frame = cap.read()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            if not ok:
                break

    def read(self):
        with self.lock:
            return self.ok, self.frame

    def release(self):
        self.running = False
        self.cap.release()


def init_game():
    """Initializes the game window and camera."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"CRITICAL ERROR: Could not load face cascade from '{CASCADE_PATH}'")
        return None, None, None, None

    # First index that actually delivers a frame. Index 0 is not always a camera
    # (on Linux it can be e.g. a hardware video decoder), so try a few.
    backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
    for index in range(5):
        cap = cv2.VideoCapture(index, backend)
        # Attempt HD resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened() and cap.read()[0]:
            break
        cap.release()
    else:
        print("CRITICAL ERROR: Could not open video camera.")
        return None, None, None, None

    cap = Camera(cap)
    ret, frame = cap.read()
    if not ret:
        return None, None, None, None

    screen_height, screen_width = frame.shape[:2]
    scale_to_frame(screen_height)
    cv2.namedWindow('Face Breaker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face Breaker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    create_blocks(screen_width, screen_height)
    spawn_ball(screen_width, screen_height)

    return cap, face_cascade, screen_width, screen_height

# --- TRACKING LOGIC (PRESERVED) ---
def track_faces(new_faces, current_time):
    global players, pending_faces, next_player_id, next_pending_id
    matched_players = {}
    new_pending_faces = {}
    unmatched_new_faces = list(new_faces)
    factor = TRACKING_SMOOTHING_FACTOR

    # 1. Match Active Players
    for player_id, player in players.items():
        best_match_index = -1
        min_distance_sq = MAX_DISTANCE_SQUARED
        for i, (x, y, w, h) in enumerate(unmatched_new_faces):
            p_x, p_y, p_w, p_h = player['face_rect']
            p_center = (p_x + p_w / 2, p_y + p_h / 2)
            f_center = (x + w / 2, y + h / 2)
            dist = (p_center[0] - f_center[0])**2 + (p_center[1] - f_center[1])**2
            if dist < min_distance_sq:
                min_distance_sq = dist
                best_match_index = i

        if best_match_index != -1:
            new_rect = unmatched_new_faces.pop(best_match_index)
            # Smooth
            o_x, o_y, o_w, o_h = player['face_rect']
            n_x, n_y, n_w, n_h = new_rect
            player['face_rect'] = (
                int(o_x * (1 - factor) + n_x * factor),
                int(o_y * (1 - factor) + n_y * factor),
                int(o_w * (1 - factor) + n_w * factor),
                int(o_h * (1 - factor) + n_h * factor)
            )
            player['last_seen_time'] = current_time
            matched_players[player_id] = player

    # 2. Match Pending
    for temp_id, pending in pending_faces.items():
        best_match_index = -1
        min_distance_sq = MAX_DISTANCE_SQUARED
        for i, (x, y, w, h) in enumerate(unmatched_new_faces):
            p_x, p_y, p_w, p_h = pending['face_rect']
            p_center = (p_x + p_w / 2, p_y + p_h / 2)
            f_center = (x + w / 2, y + h / 2)
            dist = (p_center[0] - f_center[0])**2 + (p_center[1] - f_center[1])**2
            if dist < min_distance_sq:
                min_distance_sq = dist
                best_match_index = i

        if best_match_index != -1:
            new_rect = unmatched_new_faces.pop(best_match_index)
            pending['count'] += 1
            o_x, o_y, o_w, o_h = pending['face_rect']
            n_x, n_y, n_w, n_h = new_rect
            pending['face_rect'] = (
                int(o_x * (1 - factor) + n_x * factor),
                int(o_y * (1 - factor) + n_y * factor),
                int(o_w * (1 - factor) + n_w * factor),
                int(o_h * (1 - factor) + n_h * factor)
            )
            pending['last_seen_time'] = current_time

            if pending['count'] >= MIN_CONSECUTIVE_DETECTIONS:
                # Promote to Player
                matched_players[next_player_id] = {
                    'score': 0,
                    'last_seen_time': current_time,
                    'face_rect': pending['face_rect'],
                    'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                }
                next_player_id += 1
            else:
                new_pending_faces[temp_id] = pending
        elif current_time - pending['last_seen_time'] < DECAY_TIMEOUT:
            new_pending_faces[temp_id] = pending

    # 3. New Unknowns
    for x, y, w, h in unmatched_new_faces:
        new_pending_faces[next_pending_id] = {
            'face_rect': (x, y, w, h),
            'count': 1,
            'last_seen_time': current_time
        }
        next_pending_id += 1

    # 4. Decay Active
    final_players = matched_players
    for pid, p in players.items():
        if pid not in matched_players and current_time - p['last_seen_time'] < DECAY_TIMEOUT:
            final_players[pid] = p

    players = final_players
    pending_faces = new_pending_faces

def check_aabb_collision(ball_x, ball_y, ball_r, rect_x, rect_y, rect_w, rect_h):
    """
    Simple AABB collision detection.
    Returns: None, or 'x' (horizontal hit), or 'y' (vertical hit)
    """
    # Find closest point on rect to circle center
    closest_x = max(rect_x, min(ball_x, rect_x + rect_w))
    closest_y = max(rect_y, min(ball_y, rect_y + rect_h))

    distance_x = ball_x - closest_x
    distance_y = ball_y - closest_y
    distance_sq = (distance_x ** 2) + (distance_y ** 2)

    if distance_sq < (ball_r ** 2):
        # Determine hit side based on overlap
        overlap_x = (ball_r) - abs(distance_x)
        overlap_y = (ball_r) - abs(distance_y)

        # If overlap_x is smaller, it likely hit the side.
        # However, for blocks, we mostly care about top/bottom unless hitting exact edge
        if overlap_x < overlap_y:
            return 'x'
        else:
            return 'y'
    return None

def main_game_loop():
    global game_score, blocks, balls

    cap, face_cascade, screen_width, screen_height = init_game()
    if cap is None: return

    last_activity_time = time.time()
    last_frame_time = last_activity_time

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect on a downscaled copy (4x cheaper), then map the rects back up
        small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
        min_size = max(20, int(screen_height * 0.05 * DETECT_SCALE))
        new_faces = [(int(x / DETECT_SCALE), int(y / DETECT_SCALE),
                      int(w / DETECT_SCALE), int(h / DETECT_SCALE))
                     for x, y, w, h in face_cascade.detectMultiScale(
                         small, 1.2, 5, minSize=(min_size, min_size))]
        current_time = time.time()

        # Track
        track_faces(new_faces, current_time)

        # Activity Monitor
        if len(players) > 0:
            last_activity_time = current_time
        elif current_time - last_activity_time > INACTIVITY_TIMEOUT_SEC:
            restart_game(screen_width, screen_height)
            last_activity_time = current_time

        # --- GAME LOGIC ---

        field_left, field_right = playfield(screen_width)
        paddle_y = screen_height - PADDLE_Y_OFFSET

        # 1. Place every paddle under its player's face, once per frame
        for player in players.values():
            fx, fy, fw, fh = player['face_rect']
            paddle_x = fx + fw // 2 - PADDLE_WIDTH // 2
            paddle_x = max(field_left, min(field_right - PADDLE_WIDTH, paddle_x))
            player['paddle_rect'] = (paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)

        # 2. Update Ball Positions
        # Velocities are per second, so the ball moves at the same speed no
        # matter how fast the machine renders. A hitch longer than a tenth of a
        # second is clamped, since a step that big could skip past a paddle;
        # below ~10 fps the ball goes slow-motion rather than through things.
        dt = min(current_time - last_frame_time, 0.1)
        last_frame_time = current_time

        for ball in balls[:]:
            ball['pos'][0] += ball['vel'][0] * dt
            ball['pos'][1] += ball['vel'][1] * dt

            bx, by = ball['pos']
            br = ball['radius']

            # Wall Collisions
            if bx - br <= field_left:
                ball['pos'][0] = field_left + br
                ball['vel'][0] *= -1
            elif bx + br >= field_right:
                ball['pos'][0] = field_right - br
                ball['vel'][0] *= -1

            if by - br <= 0:
                ball['pos'][1] = br
                ball['vel'][1] *= -1

            # Floor Collision (Ball Lost)
            if by - br >= screen_height:
                balls.remove(ball)
                # Respawn logic if all balls lost
                if not balls:
                    spawn_ball(screen_width, screen_height)
                continue

            # Block Collisions
            hit_block = False
            for block in blocks:
                if not block['active']: continue

                bx, by, bw, bh = block['rect']
                col_type = check_aabb_collision(ball['pos'][0], ball['pos'][1], br, bx, by, bw, bh)

                if col_type:
                    block['active'] = False
                    game_score += POINTS_PER_BLOCK
                    # Credit whoever last hit this ball
                    owner = players.get(ball.get('owner'))
                    if owner:
                        owner['score'] = owner.get('score', 0) + POINTS_PER_BLOCK
                    hit_block = True

                    if col_type == 'y':
                        ball['vel'][1] *= -1
                    else:
                        ball['vel'][0] *= -1
                    break # Only hit one block per frame per ball usually

            if hit_block: continue

            # Paddle Collisions (Players)
            for pid, player in players.items():
                paddle_x, paddle_y, _, _ = player['paddle_rect']

                # Collision Check
                col_type = check_aabb_collision(ball['pos'][0], ball['pos'][1], br,
                                                paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)

                if col_type:
                    # Determine hit position relative to paddle center (-1 to 1)
                    paddle_center = paddle_x + PADDLE_WIDTH / 2
                    hit_pos = (ball['pos'][0] - paddle_center) / (PADDLE_WIDTH / 2)
                    ball['owner'] = pid

                    # Reflect Y (always bounce up)
                    ball['vel'][1] = -abs(ball['vel'][1])

                    # Add "English" (spin/angle change) based on hit position
                    # This makes the game controllable
                    current_speed = math.sqrt(ball['vel'][0]**2 + ball['vel'][1]**2)

                    # Adjust X velocity based on where it hit
                    new_vel_x = ball['vel'][0] + (hit_pos * current_speed * 0.5)

                    # Keep the bounce clear of horizontal, or the ball spends an
                    # age skimming between the walls. Upward is a negative angle.
                    angle = math.atan2(ball['vel'][1], new_vel_x)
                    limit = math.radians(MIN_BOUNCE_ANGLE)
                    angle = max(-math.pi + limit, min(-limit, angle))

                    ball['vel'][0] = current_speed * math.cos(angle)
                    ball['vel'][1] = current_speed * math.sin(angle)

                    # Prevent ball from getting stuck in paddle
                    ball['pos'][1] = paddle_y - br - 1
                    break

        # Check Win Condition
        active_blocks = [b for b in blocks if b['active']]
        if not active_blocks:
            cv2.putText(frame, "YOU WIN!", (screen_width//2 - 200, screen_height//2),
                        cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 5)
            cv2.imshow('Face Breaker', frame)
            cv2.waitKey(3000)
            restart_game(screen_width, screen_height)

        # --- DRAWING ---

        # Dim what is out of play, so it reads as the edge of the court
        frame[:, :field_left] = (frame[:, :field_left] * 0.45).astype(np.uint8)
        frame[:, field_right:] = (frame[:, field_right:] * 0.45).astype(np.uint8)
        cv2.line(frame, (field_left, 0), (field_left, screen_height), (255, 255, 255), 3)
        cv2.line(frame, (field_right, 0), (field_right, screen_height), (255, 255, 255), 3)

        # Draw Blocks
        for block in blocks:
            if block['active']:
                bx, by, bw, bh = block['rect']
                cv2.rectangle(frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), block['color'], -1)
                cv2.rectangle(frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0,0,0), 1)

        # Draw Paddles and Connectors
        for pid, player in players.items():
            if 'paddle_rect' in player:
                px, py, pw, ph = player['paddle_rect']
                fx, fy, fw, fh = player['face_rect']
                face_center_x = fx + fw // 2
                face_center_y = fy + fh // 2

                color = player.get('color', (0, 255, 0))

                # Draw Line from Face to Paddle (Visual Feedback)
                cv2.line(frame, (face_center_x, face_center_y), (int(px + pw/2), py), color, 2)

                # Draw Face Box
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)

                # Draw Paddle
                cv2.rectangle(frame, (int(px), int(py)), (int(px+pw), int(py+ph)), color, -1)
                cv2.rectangle(frame, (int(px), int(py)), (int(px+pw), int(py+ph)), (255,255,255), 2)

                label = f"P{pid}  {player.get('score', 0)}"
                cv2.putText(frame, label, (int(px), int(py)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Balls
        for ball in balls:
            bx = int(ball['pos'][0])
            by = int(ball['pos'][1])
            cv2.circle(frame, (bx, by), int(ball['radius']), ball['color'], -1)
            cv2.circle(frame, (bx, by), int(ball['radius']), (0,0,0), 1)

        # UI
        cv2.putText(frame, f"SCORE: {game_score}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "R=Restart | Q=Quit", (screen_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Face Breaker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            restart_game(screen_width, screen_height)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_game_loop()