import cv2
import numpy as np
import random
import time
import os
import sys

# --- CONFIGURATION ---

# IMPORTANT: Make sure this path is correct!
CASCADE_PATH = r'./haarcascade_frontalface_default.xml'

# Game settings
BALL_RADIUS = 12
BALL_SPEED = 15
MAX_BALLS = 3
PADDLE_WIDTH = 120
PADDLE_HEIGHT = 20
PADDLE_Y_OFFSET = 50 # Distance from bottom
BLOCK_ROWS = 6
BLOCK_COLS = 10
BLOCK_HEIGHT = 30
POINTS_PER_BLOCK = 10

# Auto-Restart Configuration
INACTIVITY_TIMEOUT_SEC = 60.0
RESTART_ON_WIN = True

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

def restart_game():
    """Restarts the current Python script process."""
    print("--- GAME RESTARTING ---")
    os.execv(sys.executable, ['python'] + sys.argv)

def create_blocks(screen_width, screen_height):
    """Generates the grid of blocks."""
    global blocks
    blocks = []

    # Calculate margin to center blocks
    total_block_width = screen_width // BLOCK_COLS

    colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 127, 255), (139, 0, 255)]

    for row in range(BLOCK_ROWS):
        color = colors[row % len(colors)]
        for col in range(BLOCK_COLS):
            b_x = col * total_block_width
            b_y = row * BLOCK_HEIGHT + 60 # 60px buffer from top

            # Add a small gap between blocks
            blocks.append({
                'rect': [b_x + 2, b_y + 2, total_block_width - 4, BLOCK_HEIGHT - 4],
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
        vel_x = BALL_SPEED * math.cos(angle)
        vel_y = BALL_SPEED * math.sin(angle)

        new_ball = {
            'pos': [start_x, start_y],
            'vel': [vel_x, vel_y],
            'radius': BALL_RADIUS,
            'color': (255, 255, 255)
        }
        balls.append(new_ball)

def init_game():
    """Initializes the game window and camera."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"CRITICAL ERROR: Could not load face cascade from '{CASCADE_PATH}'")
        return None, None, None, None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not open video camera.")
        return None, None, None, None

    # Attempt HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        return None, None, None, None

    screen_height, screen_width = frame.shape[:2]
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

import math

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

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect
        new_faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        current_time = time.time()

        # Track
        track_faces(new_faces, current_time)

        # Activity Monitor
        if len(players) > 0:
            last_activity_time = current_time
        elif current_time - last_activity_time > INACTIVITY_TIMEOUT_SEC:
            restart_game()

        # --- GAME LOGIC ---

        # 1. Update Ball Positions
        for ball in balls[:]:
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]

            bx, by = ball['pos']
            br = ball['radius']

            # Wall Collisions
            if bx - br <= 0:
                ball['pos'][0] = br
                ball['vel'][0] *= -1
            elif bx + br >= screen_width:
                ball['pos'][0] = screen_width - br
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
                    hit_block = True

                    if col_type == 'y':
                        ball['vel'][1] *= -1
                    else:
                        ball['vel'][0] *= -1
                    break # Only hit one block per frame per ball usually

            if hit_block: continue

            # Paddle Collisions (Players)
            paddle_y = screen_height - PADDLE_Y_OFFSET

            for pid, player in players.items():
                fx, fy, fw, fh = player['face_rect']

                # Map face X to Paddle X
                # Center of face controls center of paddle
                face_center_x = fx + fw // 2
                paddle_x = face_center_x - (PADDLE_WIDTH // 2)

                # Constrain to screen
                paddle_x = max(0, min(screen_width - PADDLE_WIDTH, paddle_x))

                # Store paddle pos in player object for drawing later
                player['paddle_rect'] = (paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)

                # Collision Check
                col_type = check_aabb_collision(ball['pos'][0], ball['pos'][1], br,
                                                paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)

                if col_type:
                    # Determine hit position relative to paddle center (-1 to 1)
                    paddle_center = paddle_x + PADDLE_WIDTH / 2
                    hit_pos = (ball['pos'][0] - paddle_center) / (PADDLE_WIDTH / 2)

                    # Reflect Y (always bounce up)
                    ball['vel'][1] = -abs(ball['vel'][1])

                    # Add "English" (spin/angle change) based on hit position
                    # This makes the game controllable
                    current_speed = math.sqrt(ball['vel'][0]**2 + ball['vel'][1]**2)

                    # Adjust X velocity based on where it hit
                    new_vel_x = ball['vel'][0] + (hit_pos * 5)

                    # Normalize speed to keep it constant
                    angle = math.atan2(ball['vel'][1], new_vel_x)
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
            restart_game()

        # --- DRAWING ---

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

                cv2.putText(frame, f"P{pid}", (int(px), int(py)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
            restart_game()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_game_loop()