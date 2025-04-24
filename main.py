import os
from dotenv import load_dotenv

load_dotenv()

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import cv2
import pygame
import math
import threading
from pyfirmata2 import Arduino

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import hands as mp_hands


cam = cv2.VideoCapture(
    int(os.getenv("VIDEOCAPTURE_INDEX") or input("Camera (0, 1, 2, ...): ")),
    cv2.CAP_DSHOW,
)

if cam.isOpened():
    rval, frame = cam.read()
else:
    rval = False
    print("No camera detected")
    exit()


print(frame.shape[0], frame.shape[1])

pygame.init()
screen = pygame.display.set_mode(frame.shape[1::-1])
pygame.display.set_caption("camera preview")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

hands = mp_hands.Hands(max_num_hands=1)

avg_x_pixel, avg_y_pixel = 0, 0
avg_x_pixel_eased, avg_y_pixel_eased = 0, 0

settings = {
    "paused": False,
    "draw_elements": False,
    "draw_hand": False,
}


def track_hands():
    global hand_track, avg_x_pixel, avg_y_pixel, avg_x_pixel_eased, avg_y_pixel_eased
    hand_track = hands.process(frame)

    if hand_track.multi_hand_landmarks:
        for landmarks in hand_track.multi_hand_landmarks:
            if settings["draw_hand"]:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            landmark_indices = [0, 1, 5, 9, 13, 17]
            selected_landmarks = [landmarks.landmark[i] for i in landmark_indices]

            # Calculate the average position of the selected landmarks
            avg_x = max(
                0,
                min(
                    sum([lm.x for lm in selected_landmarks]) / len(selected_landmarks),
                    1,
                ),
            )
            avg_y = max(
                0,
                min(
                    sum([lm.y for lm in selected_landmarks]) / len(selected_landmarks),
                    1,
                ),
            )

            avg_x_pixel = int(avg_x * frame.shape[1])
            avg_y_pixel = int(avg_y * frame.shape[0])


elevation = elevation_eased = last_hand_size = hand_size = 0


def get_hand_size():
    if hand_track.multi_hand_landmarks:
        for landmark in hand_track.multi_hand_landmarks:
            selected_landmarks = [landmark.landmark[i] for i in [0, 1, 5, 9, 13, 17]]

            size = 0
            for i in range(len(selected_landmarks) - 1):
                x1, y1 = (
                    selected_landmarks[i].x * frame.shape[1],
                    selected_landmarks[i].y * frame.shape[0],
                )
                x2, y2 = (
                    selected_landmarks[i + 1].x * frame.shape[1],
                    selected_landmarks[i + 1].y * frame.shape[0],
                )
                size += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            canvas_diagonal = math.sqrt(frame.shape[1] ** 2 + frame.shape[0] ** 2)
            return size / canvas_diagonal

    return hand_size


def draw_avg():
    pygame.draw.line(
        screen,
        0x00FF00,
        (avg_x_pixel_eased, 0),
        (avg_x_pixel_eased, screen.get_height()),
    )
    pygame.draw.line(
        screen,
        0xFF0000,
        (0, avg_y_pixel_eased),
        (screen.get_width(), avg_y_pixel_eased),
    )

    rect_size = ((1 - elevation_eased) * 30) + 20
    pygame.draw.rect(
        screen,
        (
            0x00FFFF
            if rect_size == 50
            else 0x0000FF if rect_size == 20 else (255, (1 - elevation_eased) * 255, 0)
        ),
        (
            avg_x_pixel_eased - (rect_size / 2),
            avg_y_pixel_eased - (rect_size / 2),
            rect_size,
            rect_size,
        ),
        4,
    )

    rect2_size = ((hand_size) * 100 * 3, (hand_size) * 100 * 4)
    pygame.draw.rect(
        screen,
        0xFFFF00 if is_fist else 0xFF00FF,
        (
            avg_x_pixel - (rect2_size[0] / 2),
            avg_y_pixel - (rect2_size[1] / 2),
            rect2_size[0],
            rect2_size[1],
        ),
        2 if is_fist else 1,
    )


def draw_debug_text():
    debug_text_contets = [
        f"Base: {base_pin.read():.2f}'",
        f"Axle 1a: {axle1_pin.read():.2f}'",
        # f"Axle 1b: {axle1b_pin.read():.2f}'",
        f"Axle 2: {axle2_pin.read():.2f}'",
        f"Claw: {claw_pin.read():.2f}'",
    ]

    for i, c in enumerate(debug_text_contets):
        text = font.render(
            c,
            True,
            (255, 255, 255),
        )
        screen.blit(text, (10, 10 + i * 16))


def draw_elements():
    draw_avg()
    draw_debug_text()


def get_is_fist():
    if hand_track.multi_hand_landmarks:
        for landmarks in hand_track.multi_hand_landmarks:
            wrist = landmarks.landmark[0]

            fingers = [landmarks.landmark[i] for i in [8, 12, 16, 20]]
            knuckles = [landmarks.landmark[i] for i in [5, 9, 13, 17]]

            # Check if each finger landmark is below its corresponding knuckle relative to the wrist
            fist_detected = all(
                finger.y > knuckle.y if wrist.y > knuckle.y else finger.y < knuckle.y
                for finger, knuckle in zip(fingers, knuckles)
            )

            return fist_detected

    return False


is_fist = False

# Arduino logic


board = Arduino(os.getenv("ARDUINO_PORT"))

base_pin = board.get_pin(f"d:{os.getenv("BASE_PIN")}:s")
axle1_pin = board.get_pin(f"d:{os.getenv("AXLE1_PIN")}:s")
axle2_pin = board.get_pin(f"d:{os.getenv("AXLE2_PIN")}:s")
claw_pin = board.get_pin(f"d:{os.getenv("CLAW_PIN")}:s")

running = True


def move_servos():
    while running:
        if settings["paused"]:
            continue
        # base
        percentage = avg_x_pixel_eased / screen.get_width()
        base_pin.write((percentage) * 180)

        # axle 1a
        percentage = avg_y_pixel_eased / screen.get_height()
        axle1_pin.write(((1 - percentage) * 120))

        # axle 2
        axle2_pin.write(elevation_eased * 180)

        # claw
        claw_pin.write(0 if is_fist else 180)


servo_thread = threading.Thread(target=move_servos)
servo_thread.start()


while running and rval:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                settings["paused"] = not settings["paused"]
                pygame.display.set_caption(
                    "PAUSED" if settings["paused"] else "camera preview"
                )
            if event.key == pygame.K_d:
                settings["draw_elements"] = not settings["draw_elements"]
            if event.key == pygame.K_h:
                settings["draw_hand"] = not settings["draw_hand"]
    screen.fill(0x000000)

    rval, frame = cam.read()
    if not rval:
        running = False
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    if not settings["paused"]:
        track_hands()
        is_fist = get_is_fist()
        avg_x_pixel_eased += 0.04 * (avg_x_pixel - avg_x_pixel_eased)
        avg_y_pixel_eased += 0.05 * (avg_y_pixel - avg_y_pixel_eased)

        hand_size = get_hand_size()
        elevation = min(max((hand_size - 0.2) / 0.1, 0), 1)
        elevation_eased += 0.05 * (elevation - elevation_eased)

    img = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
    screen.blit(img, (0, 0))

    if settings["draw_elements"]:
        draw_elements()

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
cam.release()
servo_thread.join()
