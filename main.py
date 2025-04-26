import os
from dotenv import load_dotenv
import cv2
import pygame
import math
import threading
import time

load_dotenv()
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# Load pyfirmata2 with graceful fallback if library is not available
# This allows the app to run in simulation mode without the Arduino hardware
try:
    from pyfirmata2 import Arduino
    PYFIRMAta_AVAILABLE = True
except ImportError:
    print("WARNING: pyfirmata2 not found. Arduino control will be disabled.")
    PYFIRMAta_AVAILABLE = False
    # Define dummy classes to maintain code structure even without hardware
    class Arduino:
        def __init__(self, port): pass
        def get_pin(self, pin_def): return DummyPin()
        def exit(self): pass
        def iterate(self): pass
    class DummyPin:
        def read(self): return 0.0
        def write(self, val): pass

# MediaPipe imports for hand tracking functionality
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import hands as mp_hands


# --- Configuration Section ---
CAM_INDEX = 0  # Camera device index (0 is usually the built-in webcam)
ARDUINO_PORT = "COM39"  # Serial port where Arduino is connected - MUST BE CHANGED FOR YOUR SETUP
# Servo configuration - defines the motion range for each servo in degrees
SERVO_RANGES = {
    'base': 180,      # Full rotation range for base servo
    'axle1a': 120,    # Limited range for first joint
    'axle2': 180,     # Full range for second joint
}
# Servo pin assignments on the Arduino using Firmata pin notation
SERVO_PINS = {
    'base': "d:9:s",    # Digital pin 9, servo mode
    'axle1a': "d:10:s", # Digital pin 10, servo mode
    'axle2': "d:6:s",   # Digital pin 6, servo mode
    'claw': "d:3:s",    # Digital pin 3, servo mode for gripper
}
CLAW_ANGLES = {'open': 90, 'closed': 3}  # Servo angles for claw positions
HAND_CONFIDENCE = 0.6       # Minimum confidence threshold for hand detection
TRACKING_CONFIDENCE = 0.5   # Minimum confidence for hand tracking between frames
EASING_FACTOR_POS = 0.1     # Controls smoothing of position movements (lower = smoother but slower)
EASING_FACTOR_ELEV = 0.08   # Controls smoothing of elevation movements
FIRMATA_HANDSHAKE_WAIT = 2  # Seconds to wait for Arduino communication to establish
SERVO_UPDATE_DELAY = 20     # Milliseconds between servo position updates (~50Hz)

# --- Camera Initialization ---
print(f"Attempting to open camera index: {CAM_INDEX}")
cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW optimizes for Windows DirectShow

# Validate camera connection
if not cam.isOpened():
    print(f"Error: Could not open camera index {CAM_INDEX}")
    exit()

# Get initial frame to determine dimensions
rval, frame_test = cam.read()
if not rval:
    print("Error: Could not read initial frame from camera.")
    cam.release()
    exit()
else:
    H, W = frame_test.shape[:2]  # Get height and width from frame
    print(f"Camera opened successfully. Frame dimensions (HxW): {H}x{W}")

# --- Pygame Setup for UI Display ---
pygame.init()
screen = pygame.display.set_mode((W, H))  # Match window size to camera dimensions
pygame.display.set_caption("camera preview")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)  # Font for debug information

# --- MediaPipe Hands Setup for hand tracking ---
print("Initializing MediaPipe Hands...")
hands = mp_hands.Hands(
    static_image_mode=False,  # Video mode (not static images)
    max_num_hands=1,          # Track only one hand for simplicity
    min_detection_confidence=HAND_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE
)
print("MediaPipe Hands initialized.")

# --- Global Variables for state tracking ---
hand_landmarks_global = None  # Will store the detected hand landmarks
avg_x_pixel, avg_y_pixel = W // 2, H // 2  # Initial position at center of frame
avg_x_pixel_eased, avg_y_pixel_eased = W // 2, H // 2  # Smoothed position values
elevation = elevation_eased = hand_size = 0.0  # Z-axis control from hand size
is_fist = None  # Hand gesture state: None = no hand, True = fist, False = open hand
running = True  # Main loop control flag
last_servo_cmds = {}  # Stores most recent servo commands for debugging

# UI and control settings
settings = {
    "paused": False,       # Pause tracking when True
    "draw_elements": True, # Show UI elements when True
    "draw_hand": True,     # Show hand tracking visualization when True
}

# --- Arduino Setup ---
board = None
pins = {}  # Dictionary to hold servo pin objects
arduino_connected = False

# Try to connect to Arduino if the library is available
if PYFIRMAta_AVAILABLE:
    try:
        print(f"Attempting to connect to Arduino on {ARDUINO_PORT}...")
        board = Arduino(ARDUINO_PORT)
        print(f"Waiting {FIRMATA_HANDSHAKE_WAIT}s for Firmata handshake...")
        time.sleep(FIRMATA_HANDSHAKE_WAIT)  # Allow time for connection to establish
        # Initialize servo pins
        print("Getting pins...")
        pins['base'] = board.get_pin(SERVO_PINS['base'])
        pins['axle1a'] = board.get_pin(SERVO_PINS['axle1a'])
        pins['axle2'] = board.get_pin(SERVO_PINS['axle2'])
        pins['claw'] = board.get_pin(SERVO_PINS['claw'])
        arduino_connected = True
        print("Arduino connected successfully and pins assigned.")
    except Exception as e:
        print(f"ERROR: Could not connect to Arduino on {ARDUINO_PORT}: {e}")
        board = None
        arduino_connected = False
else:
     print("Running without Arduino control (pyfirmata2 not available).")

# Create dummy pins if Arduino connection failed or library missing
# This allows the rest of the code to work in simulation mode
if not arduino_connected:
    class DummyPin:
        def read(self): return 0.0
        def write(self, val): pass
    # Fill pins dictionary with dummy objects
    for name in SERVO_PINS:
        if name not in pins: pins[name] = DummyPin()

# --- Hand Tracking Function ---
def process_hand_landmarks(rgb_frame):
    """
    Processes an RGB frame with MediaPipe Hands and returns landmarks.
    
    Args:
        rgb_frame: RGB image frame from camera
        
    Returns:
        Hand landmarks object or None if no hand detected
    """
    rgb_frame.flags.writeable = False  # Performance optimization
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]  # Return first hand only
    return None

# --- Calculations based on landmarks ---
def calculate_hand_data(landmarks):
    """
    Calculates average hand position, size, and elevation from landmarks.
    
    Updates global variables for hand position and elevation based on
    the detected hand landmarks.
    
    Args:
        landmarks: MediaPipe hand landmarks object
    """
    global avg_x_pixel, avg_y_pixel, hand_size, elevation
    if not landmarks: return

    # Calculate average position using wrist and finger base joints
    # This provides a stable center point for the hand
    landmark_indices = [
        mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP ]
    selected_landmarks = [landmarks.landmark[i] for i in landmark_indices]
    avg_x = sum(lm.x for lm in selected_landmarks) / len(selected_landmarks)
    avg_y = sum(lm.y for lm in selected_landmarks) / len(selected_landmarks)
    # Ensure values stay within valid range
    avg_x = max(0.0, min(1.0, avg_x))
    avg_y = max(0.0, min(1.0, avg_y))
    # Convert normalized coordinates to pixel values
    avg_x_pixel = int(avg_x * W)
    avg_y_pixel = int(avg_y * H)

    # Calculate hand size based on distance between wrist and middle finger base
    # This gives us a measure of how close the hand is to the camera (Z-axis control)
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    x1, y1 = wrist.x * W, wrist.y * H
    x2, y2 = middle_mcp.x * W, middle_mcp.y * H
    pixel_dist = math.hypot(x2 - x1, y2 - y1)  # Efficient Euclidean distance
    canvas_diagonal = math.hypot(W, H)
    normalized_size = pixel_dist / canvas_diagonal if canvas_diagonal else 0

    # Map normalized size to elevation with scaling and constraints
    hand_size = normalized_size * 5  # Scale up for better sensitivity
    elevation_raw = (hand_size - 0.4) / 0.6  # Map to usable range
    elevation = min(max(elevation_raw, 0.0), 1.0)  # Clamp to 0-1 range

# --- Drawing Functions ---
def draw_avg():
    """
    Draws crosshairs and visual indicators for hand position and state.
    Shows the target position, elevation level, and grip state.
    """
    # Draw crosshairs showing X and Y positioning
    pygame.draw.line(screen, 0x00FF00, (int(avg_x_pixel_eased), 0), (int(avg_x_pixel_eased), H))
    pygame.draw.line(screen, 0xFF0000, (0, int(avg_y_pixel_eased)), (W, int(avg_y_pixel_eased)))

    # Draw elevation indicator rectangle - size changes with Z-axis position
    rect_size = max(10, (1 - elevation_eased) * 40 + 10)
    pygame.draw.rect(
        screen, 0x00FFFF,
        pygame.Rect(int(avg_x_pixel_eased - rect_size / 2), int(avg_y_pixel_eased - rect_size / 2),
                    int(rect_size), int(rect_size)), 4)

    # Draw fist indicator rectangle - shows whether claw should be open or closed
    if is_fist is not None:  # Only draw if hand is detected
        rect_width, rect_height = 50, 70
        fist_color = 0xFFFF00 if is_fist else 0xFF00FF  # Yellow for closed, Pink for open
        fist_border = 3 if is_fist else 1  # Thicker border when fist is detected
        pygame.draw.rect(
            screen, fist_color,
            pygame.Rect(int(avg_x_pixel_eased - rect_width / 2), int(avg_y_pixel_eased - rect_height / 2),
                        rect_width, rect_height),
            fist_border, border_radius=5)

def draw_debug_text():
    """
    Displays debug information on the screen.
    Shows FPS, tracking status, and current servo commands.
    """
    debug_lines = [
        f"FPS: {clock.get_fps():.1f}",
        f"Hand Detected: {'Yes' if hand_landmarks_global else 'No'}",
        f"Paused: {settings['paused']}",
        f"Arduino: {'Connected' if arduino_connected else 'N/A'}",
        f"Elevation: {elevation_eased:.2f}",
        f"Fist: {is_fist}",
        f"Avg Pos: ({int(avg_x_pixel_eased)}, {int(avg_y_pixel_eased)})",
    ]
    # Add servo commands if Arduino is connected
    if arduino_connected:
         debug_lines.extend([
            f"Base Cmd: {last_servo_cmds.get('base', 'N/A')}",
            f"Axle1a Cmd: {last_servo_cmds.get('axle1a', 'N/A')}",
            f"Axle2 Cmd: {last_servo_cmds.get('axle2', 'N/A')}",
            f"Claw Cmd: {last_servo_cmds.get('claw', 'N/A')}",
        ])

    # Render and display each line of text
    for i, line in enumerate(debug_lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, 10 + i * 18))

def draw_elements():
    """Calls all drawing functions for UI overlays."""
    draw_avg()
    draw_debug_text()

# --- Gesture Detection Functions ---
def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx):
    """
    Checks if a finger is extended based on tip vs PIP y-coordinate.
    
    A finger is considered extended if its tip is above (smaller y value)
    its PIP (proximal interphalangeal) joint.
    
    Args:
        landmarks: MediaPipe hand landmarks
        finger_tip_idx: Index of the fingertip landmark
        finger_pip_idx: Index of the finger PIP joint landmark
        
    Returns:
        Boolean indicating if finger is extended
    """
    if not landmarks: return False
    # Extended if tip is vertically above PIP (smaller y-coordinate)
    return landmarks.landmark[finger_tip_idx].y < landmarks.landmark[finger_pip_idx].y

def get_is_fist(landmarks):
    """
    Checks if the hand is likely in a fist position (fingers curled).
    
    A fist is detected when all four fingers are curled inward.
    
    Args:
        landmarks: MediaPipe hand landmarks
        
    Returns:
        Boolean indicating fist detection, or None if no landmarks
    """
    if not landmarks: return None
    # Check if major fingers are curled (tip y > MCP y)
    # For each finger, compare tip position to base position
    index_curled = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_curled = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_curled = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_curled = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
    
    return index_curled and middle_curled and ring_curled and pinky_curled

def is_middle_finger_up(landmarks):
    """
    Checks for the middle finger up gesture (middle extended, others curled).
    Used as an exit gesture to close the application.
    
    Args:
        landmarks: MediaPipe hand landmarks
        
    Returns:
        Boolean indicating detection of middle finger gesture
    """
    if not landmarks: return False
    # Check if only middle finger is extended
    middle_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    index_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    # Simple thumb check
    thumb_curled = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    
    return middle_extended and index_curled and ring_curled and pinky_curled and thumb_curled

# --- Servo Control Thread ---
def move_servos():
    """
    Thread function to continuously update servo positions based on global state.
    Runs in separate thread to maintain consistent timing for servo control.
    """
    global last_servo_cmds
    print("Servo thread started.")
    while running:
        if settings["paused"] or not arduino_connected:
            pygame.time.wait(100)  # Reduce CPU usage when inactive
            continue

        try:
            # Calculate target angles based on eased values
            # Map normalized coordinates to servo angle ranges
            base_target = (avg_x_pixel_eased / W) * SERVO_RANGES['base']
            axle1a_target = (1 - (avg_y_pixel_eased / H)) * SERVO_RANGES['axle1a']  # Invert Y axis
            axle2_target = elevation_eased * SERVO_RANGES['axle2']  # Z-axis from hand size

            # Clamp angles to valid ranges and convert to integer commands
            base_cmd = max(0, min(SERVO_RANGES['base'], int(base_target)))
            axle1a_cmd = max(0, min(SERVO_RANGES['axle1a'], int(axle1a_target)))
            axle2_cmd = max(0, min(SERVO_RANGES['axle2'], int(axle2_target)))

            # Determine claw command based on fist state
            claw_cmd = CLAW_ANGLES['open']
            if is_fist is not None:  # Only set if hand is detected
                claw_cmd = CLAW_ANGLES['closed'] if is_fist else CLAW_ANGLES['open']

            # Write commands to Arduino pins
            pins['base'].write(base_cmd)
            pins['axle1a'].write(axle1a_cmd)
            pins['axle2'].write(axle2_cmd)
            pins['claw'].write(claw_cmd)

            # Store commands for debugging display
            last_servo_cmds = {
                'base': base_cmd, 'axle1a': axle1a_cmd,
                'axle2': axle2_cmd, 'claw': claw_cmd
            }

        except Exception as e:
            print(f"ERROR in servo thread write: {e}")
            pygame.time.wait(500)  # Pause briefly after an error

        # Control update rate to avoid overwhelming the Arduino
        pygame.time.wait(SERVO_UPDATE_DELAY)

    print("Servo thread finished.")

# --- Start Servo Thread ---
servo_thread = None
if arduino_connected:
    # Start servo control in separate thread to prevent blocking the main loop
    servo_thread = threading.Thread(target=move_servos, daemon=True)
    servo_thread.start()
else:
    print("Servo thread not started (Arduino not connected).")

# --- Main Game Loop ---
print("Starting main loop...")
try:
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_SPACE:
                    # Toggle pause state with spacebar
                    settings["paused"] = not settings["paused"]
                    pygame.display.set_caption("PAUSED" if settings["paused"] else "camera preview")
                if event.key == pygame.K_d: settings["draw_elements"] = not settings["draw_elements"]
                if event.key == pygame.K_h: settings["draw_hand"] = not settings["draw_hand"]

        # --- Frame Acquisition ---
        rval, frame_bgr = cam.read()
        if not rval:
            print("Error: Failed to grab frame inside loop.")
            running = False; continue

        # --- Frame Processing ---
        frame_bgr = cv2.flip(frame_bgr, 1)  # Apply mirror effect for intuitive control
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
        hand_landmarks_global = None  # Reset landmarks for this frame

        if not settings["paused"]:
            # Process hand tracking when not paused
            hand_landmarks_global = process_hand_landmarks(frame_rgb)

            if hand_landmarks_global:
                # Hand detected: calculate data and check gestures
                calculate_hand_data(hand_landmarks_global)
                is_fist = get_is_fist(hand_landmarks_global)

                # Check for exit gesture (middle finger up)
                if is_middle_finger_up(hand_landmarks_global):
                    print("Middle finger gesture detected! Closing app.")
                    running = False
                    time.sleep(0.5)  # Slight delay to prevent accidental triggering
            else:
                # No hand detected
                is_fist = None
                elevation = 0  # Reset elevation when hand disappears

            # Apply easing to movements for smoother transitions
            avg_x_pixel_eased += EASING_FACTOR_POS * (avg_x_pixel - avg_x_pixel_eased)
            avg_y_pixel_eased += EASING_FACTOR_POS * (avg_y_pixel - avg_y_pixel_eased)
            elevation_eased += EASING_FACTOR_ELEV * (elevation - elevation_eased)

        # --- Drawing ---
        screen.fill(0x000000)  # Clear screen with black

        # Draw hand landmarks overlay if enabled and hand detected
        if settings["draw_hand"] and hand_landmarks_global:
            mp_drawing.draw_landmarks(
                frame_rgb, hand_landmarks_global, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Convert processed frame for Pygame display
        frame_rgb.flags.writeable = True  # Ensure writeable for tobytes conversion
        img_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (W, H), "RGB")
        screen.blit(img_surface, (0, 0))

        # Draw UI elements (crosshairs, debug text)
        if settings["draw_elements"]:
            draw_elements()

        pygame.display.flip()  # Update the entire display
        clock.tick(60)  # Limit to 60 FPS

except KeyboardInterrupt:
    # Handle Ctrl+C shutdown
    print("\nKeyboardInterrupt detected. Exiting...")
    running = False

finally:
    # --- Cleanup Section ---
    print("Exiting main loop. Cleaning up...")
    
    # Close pygame
    pygame.quit()
    print("Pygame quit.")
    
    # Release camera resources
    if cam.isOpened():
        cam.release()
        print("Camera released.")
    cv2.destroyAllWindows()
    print("CV2 windows destroyed.")

    # Signal servo thread to stop
    running = False  # Ensure servo thread loop condition is False

    # Close Arduino connection
    if arduino_connected and board:
        print("Closing Arduino connection...")
        # Optional: Could add code here to move servos to a neutral "home" position
        try:
            board.exit()
            print("Arduino board closed.")
        except Exception as e:
            print(f"Error closing Arduino board: {e}")

    # Wait for servo thread to finish
    if servo_thread and servo_thread.is_alive():
        print("Waiting for servo thread to complete...")
        servo_thread.join(timeout=2.0)
        if servo_thread.is_alive():
            print("Warning: Servo thread did not exit cleanly.")

    print("Script finished.")