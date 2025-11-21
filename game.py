import cv2
import numpy as np
import random
import time
import os
import sys

# --- CONFIGURATION ---

# IMPORTANT: Make sure this path is correct!
# Download the file from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
CASCADE_PATH = r'./haarcascade_frontalface_default.xml'

# Game settings
GRAVITY = 0.6
BALL_RADIUS = 25
BOUNCE_STRENGTH = -18  # Strong negative velocity for upward bounce
MAX_BALLS = 8
POINTS_PER_BOUNCE = 1
SCORE_TO_ADD_BALL = 10
MAX_BALL_BOUNCES = 5 # Max bounces allowed before ball is removed

# Auto-Restart Configuration
INACTIVITY_TIMEOUT_SEC = 60.0 # Time (in seconds) with no faces detected before restart

# Player Tracking Configuration (FOR STABILITY)
TRACKING_SMOOTHING_FACTOR = 0.5 # New: Lower value means more smoothing, higher means more raw movement.
MAX_DISTANCE_SQUARED = 90000 # Max 300 pixels movement per frame before losing track of a player
DECAY_TIMEOUT = 2.0          # Reduced: Seconds before an unseen player is removed (down from 5.0s)
MIN_CONSECUTIVE_DETECTIONS = 3 # Frames required to confirm a new player

# --- GAME STATE ---
balls = []
# 'players' stores confirmed active player state: {player_id: {'score', 'last_seen_time', 'face_rect'}}
players = {}
# 'pending_faces' stores faces seen recently but not yet confirmed:
# {temp_id: {'count', 'last_seen_time', 'face_rect'}}
pending_faces = {}
next_player_id = 1
next_pending_id = 1
last_ball_spawn_score = -1

# --- HELPER FUNCTIONS ---

def restart_game():
    """Restarts the current Python script process."""
    print("--- GAME RESTARTING ---")
    print(f"Executing restart now (manual 'r' keypress or inactivity timeout).")
    os.execv(sys.executable, ['python'] + sys.argv)

def spawn_ball(screen_width):
    """Creates a new ball and adds it to the list."""
    global balls
    if len(balls) < MAX_BALLS:
        new_ball = {
            'pos': [random.randint(BALL_RADIUS * 2, screen_width - BALL_RADIUS * 2), BALL_RADIUS],
            'vel': [random.uniform(-3, 3), 0],
            'radius': BALL_RADIUS,
            'color': (random.randint(0, 255), random.randint(0, 255), 255),
            'bounces': 0 # Bounce counter
        }
        balls.append(new_ball)
        print(f"Spawning new ball! Total balls: {len(balls)}")

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

    # Attempt to set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("CRITICAL ERROR: Could not read frame from camera.")
        cap.release()
        return None, None, None, None

    screen_height, screen_width = frame.shape[:2]
    print(f"Camera feed initialized: {screen_width}x{screen_height}")

    cv2.namedWindow('Face Bouncer', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face Bouncer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    spawn_ball(screen_width)

    return cap, face_cascade, screen_width, screen_height

def track_faces(new_faces, current_time):
    """
    Implements stable tracking by matching faces to existing players (Green border)
    or temporary pending faces (Yellow border) before confirming a new player.
    """
    global players, pending_faces, next_player_id, next_pending_id

    matched_players = {}

    # new_pending_faces will hold all pending faces for the NEXT frame:
    # 1. Matched but not promoted (count > 1)
    # 2. Missed but not decayed
    # 3. Brand new faces (count = 1)
    new_pending_faces = {}
    unmatched_new_faces = list(new_faces)

    # Smoothing factor for player movement
    factor = TRACKING_SMOOTHING_FACTOR

    # --- 1. Match new faces to ACTIVE players (Green, Scored) ---
    for player_id, player in players.items():
        best_match_index = -1
        min_distance_sq = MAX_DISTANCE_SQUARED

        for i, (x, y, w, h) in enumerate(unmatched_new_faces):
            p_x, p_y, p_w, p_h = player['face_rect']
            p_center = (p_x + p_w / 2, p_y + p_h / 2)
            f_center = (x + w / 2, y + h / 2)
            distance_sq = (p_center[0] - f_center[0])**2 + (p_center[1] - f_center[1])**2

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                best_match_index = i

        if best_match_index != -1:
            # Match found for active player: Update position and remove face from pool
            new_face_rect = unmatched_new_faces.pop(best_match_index)

            # Apply Smoothing (Linear Interpolation)
            old_x, old_y, old_w, old_h = player['face_rect']
            new_x, new_y, new_w, new_h = new_face_rect

            smoothed_x = int(old_x * (1 - factor) + new_x * factor)
            smoothed_y = int(old_y * (1 - factor) + new_y * factor)
            smoothed_w = int(old_w * (1 - factor) + new_w * factor)
            smoothed_h = int(old_h * (1 - factor) + new_h * factor)

            player['face_rect'] = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)

            player['last_seen_time'] = current_time
            matched_players[player_id] = player

    # --- 2. Match remaining new faces to PENDING faces (Yellow, Unconfirmed) ---

    for temp_id, pending in pending_faces.items():
        best_match_index = -1
        min_distance_sq = MAX_DISTANCE_SQUARED

        for i, (x, y, w, h) in enumerate(unmatched_new_faces):
            p_x, p_y, p_w, p_h = pending['face_rect']
            p_center = (p_x + p_w / 2, p_y + p_h / 2)
            f_center = (x + w / 2, y + h / 2)
            distance_sq = (p_center[0] - f_center[0])**2 + (p_center[1] - f_center[1])**2

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                best_match_index = i

        if best_match_index != -1:
            # Match found for pending face: Increment count and remove face from pool
            new_rect = unmatched_new_faces.pop(best_match_index)
            pending['count'] += 1

            # Apply Smoothing for pending faces as well
            old_x, old_y, old_w, old_h = pending['face_rect']
            new_x, new_y, new_w, new_h = new_rect

            smoothed_x = int(old_x * (1 - factor) + new_x * factor)
            smoothed_y = int(old_y * (1 - factor) + new_y * factor)
            smoothed_w = int(old_w * (1 - factor) + new_w * factor)
            smoothed_h = int(old_h * (1 - factor) + new_h * factor)

            pending['face_rect'] = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
            pending['last_seen_time'] = current_time

            if pending['count'] >= MIN_CONSECUTIVE_DETECTIONS:
                # CONFIRMED: Promote to active player
                print(f"Player {next_player_id} confirmed after {pending['count']} frames.")
                new_player = {
                    'score': 0,
                    'last_seen_time': current_time,
                    'face_rect': pending['face_rect'] # Use the final smoothed position
                }
                matched_players[next_player_id] = new_player
                next_player_id += 1
            else:
                # Still pending, keep tracking (Add to the next frame's pending list)
                new_pending_faces[temp_id] = pending

        elif current_time - pending['last_seen_time'] < DECAY_TIMEOUT:
            # Pending face missed this frame, but keep them until decay timeout is hit
            new_pending_faces[temp_id] = pending

    # --- 3. Add remaining unmatched faces as new PENDING faces ---
    for x, y, w, h in unmatched_new_faces:
        new_pending = {
            'face_rect': (x, y, w, h),
            'count': 1,
            'last_seen_time': current_time
        }
        new_pending_faces[next_pending_id] = new_pending
        next_pending_id += 1


    # --- 4. Finalize State (Decay) ---

    # Active Player Decay: Keep players that were seen this frame OR haven't decayed
    final_players = matched_players
    for player_id, player in players.items():
        if player_id not in matched_players and current_time - player['last_seen_time'] < DECAY_TIMEOUT:
            # Ensure the player object is copied back if it's still within the decay window
            final_players[player_id] = player

    players = final_players

    # Pending Face Cleanup: We use the already accumulated list from steps 2 & 3.
    pending_faces = new_pending_faces


def main_game_loop():
    """Runs the main game loop."""
    global next_player_id, last_ball_spawn_score, balls

    cap, face_cascade, screen_width, screen_height = init_game()

    if cap is None:
        return

    last_activity_time = time.time()

    while True:
        # --- 1. CAPTURE FRAME AND FIND FACES ---
        ret, frame = cap.read()
        if not ret:
            print("Error: Lost camera feed.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        new_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        current_time = time.time()

        # Run stable tracking logic
        track_faces(new_faces, current_time)

        # --- ACTIVITY CHECK & AUTO-RESTART ---
        # Activity check now includes both confirmed players and faces pending confirmation
        if len(players) > 0 or len(pending_faces) > 0:
            last_activity_time = current_time

            # If no faces are detected for the timeout period, restart the game
        elif current_time - last_activity_time > INACTIVITY_TIMEOUT_SEC:
            restart_game()
            break

            # Calculate the collective score for spawning new balls
        total_score = sum(p['score'] for p in players.values())

        # --- 2. GAME LOGIC ---

        # Check for new ball spawn (Score-based increase)
        # FIX: Corrected typo from SCORE_BALL_TO_ADD to SCORE_TO_ADD_BALL
        if total_score > 0 and total_score // SCORE_TO_ADD_BALL > last_ball_spawn_score and len(balls) < MAX_BALLS:
            spawn_ball(screen_width)
            last_ball_spawn_score = total_score // SCORE_TO_ADD_BALL

        # Update and check collisions for each ball
        for ball in balls[:]:

            # Apply gravity
            ball['vel'][1] += GRAVITY

            # Update position
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]

            # --- Collision Detection ---
            ball_x = int(ball['pos'][0])
            ball_y = int(ball['pos'][1])
            ball_r = ball['radius']

            ball_consumed_by_hit = False

            # Check for face hits (paddles) - only use CONFIRMED players
            for player_id, player in players.items():
                x, y, w, h = player['face_rect']

                # Find closest point on the rect to the ball's center
                closest_x = max(x, min(ball_x, x + w))
                closest_y = max(y, min(ball_y, y + h))

                # Calculate distance
                distance = ( (ball_x - closest_x)**2 + (ball_y - closest_y)**2 )**0.5

                if distance < ball_r and ball['vel'][1] > 0: # Hit and moving downwards

                    # 1. UPDATE THIS PLAYER'S SCORE AND BALL STATE
                    player['score'] += POINTS_PER_BOUNCE
                    ball['bounces'] += 1 # Increment bounce count

                    # 2. APPLY BOUNCE PHYSICS
                    # Reverse and dampen vertical velocity
                    ball['vel'][1] = BOUNCE_STRENGTH

                    # Add small horizontal speed change based on where the ball hit the face
                    center_x = x + w / 2
                    hit_offset = ball_x - center_x
                    ball['vel'][0] += hit_offset / (w / 2) * 5 # Max +/- 5 horizontal velocity change

                    # 3. CHECK FOR MAX BOUNCES (Despawn condition)
                    if ball['bounces'] >= MAX_BALL_BOUNCES:
                        # Despawn the ball
                        balls.remove(ball)
                        ball_consumed_by_hit = True
                        print(f"Ball despawned after {MAX_BALL_BOUNCES} bounces.")

                        # FIX: Spawn a replacement ball immediately if the ball count is low.
                        if len(balls) < MAX_BALLS:
                            spawn_ball(screen_width)

                    break # Stop checking other players for this ball

            # If the ball was removed by a hit, skip the rest of the physics and screen checks
            if ball_consumed_by_hit:
                continue

                # Check for screen boundaries

            # Left/Right walls
            if ball['pos'][0] - ball_r <= 0:
                ball['pos'][0] = ball_r
                ball['vel'][0] *= -0.8
            elif ball['pos'][0] + ball_r >= screen_width:
                ball['pos'][0] = screen_width - ball_r
                ball['vel'][0] *= -0.8

            # Top wall
            if ball['pos'][1] - ball_r <= 0:
                ball['pos'][1] = ball_r
                ball['vel'][1] *= -0.5

            # Bottom wall (CONSUME BALL - MISS)
            if ball['pos'][1] + ball_r >= screen_height:

                # The ball is consumed (removed) upon hitting the floor.
                balls.remove(ball)

                # Adjust spawn tracking to make it slightly easier to get the next ball back
                last_ball_spawn_score = max(-1, last_ball_spawn_score - 1)

                # If the last ball was consumed, immediately spawn a new one to keep the game running.
                if not balls:
                    spawn_ball(screen_width)

                continue

                # --- 3. DRAWING ---

        # Draw all confirmed players (Green border, with score)
        for player_id, player in players.items():
            x, y, w, h = player['face_rect']

            # Draw the face rectangle (Paddle) - Green for active player
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

            # Draw player ID and score above the face
            score_text = f"P{player_id}: {player['score']}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10 # Position 10 pixels above the face

            # Draw white background/shadow for readability
            cv2.putText(frame, score_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 6)
            # Draw score text in blue
            cv2.putText(frame, score_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 3)

        # Draw all balls
        for ball in balls:
            bounce_count = MAX_BALL_BOUNCES - ball['bounces']
            bounce_text = f'{bounce_count}'

            # Draw the ball
            cv2.circle(frame, (int(ball['pos'][0]), int(ball['pos'][1])), ball['radius'], ball['color'], -1)
            cv2.circle(frame, (int(ball['pos'][0]), int(ball['pos'][1])), ball['radius'], (255, 255, 255), 2)

            # Draw remaining bounce count on the ball
            text_scale = 0.8
            text_thickness = 2

            # Center the text
            text_size = cv2.getTextSize(bounce_text, cv2.FONT_HERSHEY_DUPLEX, text_scale, text_thickness)[0]
            text_x = int(ball['pos'][0] - text_size[0] / 2)
            # Adjust y for centering (text_size[1] is height)
            text_y = int(ball['pos'][1] + text_size[1] / 2)

            # Draw black outline
            cv2.putText(frame, bounce_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, text_scale, (0, 0, 0), text_thickness + 2)
            # Draw white text
            cv2.putText(frame, bounce_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 255), text_thickness)

            # Draw UI (Total Score and Update/Restart Text)
        font_scale = 1.5
        font_thickness = 4

        # Total Score (Left Top)
        cv2.putText(frame, f'{total_score}', (30, 70),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), font_thickness * 2)
        cv2.putText(frame, f'{total_score}', (30, 70),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 255), font_thickness)

        # Small Update/Quit Text (Right Top)
        update_text = 'R/ESC'
        cv2.putText(frame, update_text, (screen_width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 4. DISPLAY FRAME ---
        cv2.imshow('Face Bouncer', frame)

        # --- 5. HANDLE QUIT AND MANUAL RESTART ---
        key = cv2.waitKey(1) & 0xFF

        # Quit on 'q' or ESCAPE (ASCII 27)
        if key == ord('q') or key == 27:
            print(f"Manual quit. Final Score: {total_score}")
            break

        # Manual restart/update on 'r'
        elif key == ord('r'):
            print("Manual restart/update triggered by 'r' key.")
            restart_game()
            break

            # --- 6. CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_game_loop()