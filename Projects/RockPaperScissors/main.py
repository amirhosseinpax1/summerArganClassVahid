import os
import random
import sys
import time
from typing import Optional


import cv2

try:
    import pygame
except ImportError:
    pygame = None

from rps_ml import (
    LABELS,
    FA_LABELS,
    HandFeatureExtractor,
    append_sample,
    dataset_counts,
    train_model,
    load_model,
    predict_move,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


KEY_TO_LABEL = {
    ord("r"): "rock",
    ord("p"): "paper",
    ord("s"): "scissors",
}

MOVE_TO_WIN = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}

EN_LABELS = {
    "rock": "Rock",
    "paper": "Paper",
    "scissors": "Scissors",
}


SIDE_PANEL_WIDTH = 620

SOUND_BASE_NAMES = {
    "tick": "mixkit-clock-countdown-bleeps-916",
    "user_win": "mixkit-losing-bleeps-2026",
    "system_win": "mixkit-quick-win-video-game-notification-269",
}

SOUND_EXTENSIONS = [".wav"]
SOUNDS = {}


def find_sound_file(base_name: str) -> Optional[str]:
    project_dir = os.path.dirname(os.path.abspath(__file__))

    for ext in SOUND_EXTENSIONS:
        path = os.path.join(project_dir, base_name + ext)
        if os.path.exists(path):
            return path

    return None


def init_audio():
    if pygame is None:
        print("Audio disabled: pygame is not installed. Run: pip install pygame")
        return

    try:
        pygame.mixer.init()

        for sound_key, base_name in SOUND_BASE_NAMES.items():
            path = find_sound_file(base_name)

            if path is None:
                print(f"Audio file not found for {sound_key}: {base_name}.wav / {base_name}.mp3 / {base_name}.ogg")
                continue

            SOUNDS[sound_key] = pygame.mixer.Sound(path)
            print(f"Loaded sound: {sound_key} => {os.path.basename(path)}")

    except Exception as e:
        print(f"Audio initialization failed: {e}")


def play_sound(sound_key: str):
    sound = SOUNDS.get(sound_key)

    if sound is None:
        return

    try:
        sound.play()
    except Exception as e:
        print(f"Could not play sound {sound_key}: {e}")


def fa_result(user_move: str, system_move: str) -> str:
    if user_move == system_move:
        return "Tie!"

    if MOVE_TO_WIN[user_move] == system_move:
        return "You won!"

    return "System won!"


def draw_text(frame, text, x, y, scale=0.75, thickness=2):
    """
    This function draws readable English text using OpenCV.
    All important text is now displayed on a side panel outside the camera frame.
    """
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def make_display(frame):
    """
    Creates a wider display canvas with a dark side panel.
    The camera image stays unchanged, and all text is shown outside the image area.
    """
    display = cv2.copyMakeBorder(
        frame,
        0,
        0,
        0,
        SIDE_PANEL_WIDTH,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    return display


def draw_panel_lines(display, frame_width, lines, start_y=40, line_gap=34, scale=0.65, thickness=2):
    """
    Draws multiple text lines on the side panel, outside the camera frame.
    """
    x = frame_width + 25
    y = start_y

    for line in lines:
        draw_text(display, line, x, y, scale, thickness)
        y += line_gap


def print_help():
    print("\n" + "=" * 70)
    print("Rock / Paper / Scissors project with camera and self-learning model")
    print("=" * 70)
    print("Keys:")
    print("  r  = Save current sample as Rock/Fist")
    print("  p  = Save current sample as Paper")
    print("  s  = Save current sample as Scissors")
    print("  t  = Train model with saved samples")
    print("  g  = Start game")
    print("  h  = Show help")
    print("  q  = Quit")
    print("-" * 70)
    print("How to use:")
    print("1) First, collect several samples for each gesture. For example, at least 15 samples per class.")
    print("2) Show your hand in different angles and distances, then press the correct label key.")
    print("3) Press t to train the model.")
    print("4) Press g to start the game.")
    print("=" * 70 + "\n")


def ask_correction(predicted_label: str) -> Optional[str]:
    """
    If confidence is low, the program asks the user for the correct label
    so it can improve itself.
    """
    print("\nThe model is not confident.")
    print(f"Model prediction: {EN_LABELS[predicted_label]}")
    print("What was your correct hand gesture?")
    print("  r = Rock/Fist | p = Paper | s = Scissors | Enter = prediction is correct | x = do not save")

    ans = input("Correct label: ").strip().lower()

    if ans == "":
        return predicted_label
    if ans == "r":
        return "rock"
    if ans == "p":
        return "paper"
    if ans == "s":
        return "scissors"
    if ans == "x":
        return None

    print("Invalid input. Sample was not saved.")
    return None

def play_round(cap, extractor: HandFeatureExtractor, model, scores: dict, confidence_threshold=0.65):
    """
    Runs one game round:
    - The system randomly picks its move at second 5.
    - The system move stays hidden until the end of the countdown.
    - At the end of the countdown, hand features are captured.
    - The model predicts the user's move.
    - The result is calculated and stored.
    """
    system_move = random.choice(LABELS)
    print("\nNew round started.")
    print("The system picked its move at second 5, but it will stay hidden until the end.")
    print("Get ready: 5 ... 4 ... 3 ... 2 ... 1 ...")

    captured_features = None
    captured_frame = None
    captured_landmarks = None

    # Countdown audio is 5 seconds long, so it must be played only once
    play_sound("tick")

    for number in [5, 4, 3, 2, 1]:
        start = time.time()
        while time.time() - start < 1.0:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            features, landmarks = extractor.extract(frame)
            extractor.draw(frame, landmarks)

            display = make_display(frame)
            draw_panel_lines(
                display,
                frame.shape[1],
                [
                    "RPS Self-Learning Camera",
                    "",
                    f"Countdown: {number}",
                    "Show ROCK / PAPER / SCISSORS",
                    "System already picked at 5 sec",
                    "",
                    "Press Q to quit",
                ],
                start_y=40,
                line_gap=34,
                scale=0.65,
                thickness=2,
            )

            cv2.imshow("RPS Self Learning Camera", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

            if number == 1 and features is not None:
                captured_features = features
                captured_frame = frame
                captured_landmarks = landmarks

    if captured_features is None:
        print("No hand was detected at the final moment. This round was not counted.")
        return True

    user_move, confidence, prob_map = predict_move(model, captured_features)

    print("\nModel detection:")
    print(f"  Your move: {EN_LABELS[user_move]}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Probabilities: {prob_map}")

    if confidence < confidence_threshold:
        corrected = ask_correction(user_move)
        if corrected is not None:
            append_sample(captured_features, corrected)
            user_move = corrected
            ok, train_msg = train_model()
            print(train_msg)
        else:
            print("Correction sample was not saved.")

    result = fa_result(user_move, system_move)

    scores["rounds"] += 1
    if result == "You won!":
        scores["user"] += 1
        play_sound("user_win")
    elif result == "System won!":
        scores["system"] += 1
        play_sound("system_win")
    else:
        scores["tie"] += 1

    print("\nRound result:")
    print(f"  You: {EN_LABELS[user_move]}")
    print(f"  System: {EN_LABELS[system_move]}")
    print(f"  Result: {result}")
    print(f"  Scores => You: {scores['user']} | System: {scores['system']} | Tie: {scores['tie']}")

    # Show the result for a few seconds on the side panel
    end_time = time.time() + 5
    while time.time() < end_time:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        features, landmarks = extractor.extract(frame)
        extractor.draw(frame, landmarks)

        display = make_display(frame)
        draw_panel_lines(
            display,
            frame.shape[1],
            [
                "Round Result",
                "",
                f"You: {user_move}",
                f"System: {system_move}",
                f"Result: {result}",
                "",
                f"Score You/System/Tie:",
                f"{scores['user']} / {scores['system']} / {scores['tie']}",
                "",
                "Press Q to quit",
            ],
            start_y=40,
            line_gap=34,
            scale=0.65,
            thickness=2,
        )

        cv2.imshow("RPS Self Learning Camera", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return True

def main():
    print_help()
    init_audio()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Laptop camera could not be opened. It may be used by another application.")
        return

    extractor = HandFeatureExtractor(max_num_hands=1)
    model = load_model()

    if model is None:
        print("No saved model was found. First collect samples, then press t.")
    else:
        print("Previous model loaded successfully. You can press g to play or add new samples.")

    scores = {
        "rounds": 0,
        "user": 0,
        "system": 0,
        "tie": 0,
    }

    last_message = "Collect samples: R/P/S, Train: T, Game: G, Quit: Q"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error while reading camera frame.")
            break

        frame = cv2.flip(frame, 1)

        features, landmarks = extractor.extract(frame)
        extractor.draw(frame, landmarks)

        counts = dataset_counts()

        panel_lines = [
            "RPS Self-Learning Camera",
            "",
            "Keys:",
            "R = rock",
            "P = paper",
            "S = scissors",
            "T = train",
            "G = game",
            "H = help",
            "Q = quit",
            "",
            f"Samples:",
            f"Rock / Paper / Scissors:",
            f"{counts['rock']} / {counts['paper']} / {counts['scissors']}",
            "",
            f"Status:",
            last_message[:75],
            "",
        ]

        if features is None:
            panel_lines.append("No hand detected")
        else:
            panel_lines.append("Hand detected")
            panel_lines.append("Press R/P/S to label pose")

            if model is not None:
                pred, conf, _ = predict_move(model, features)
                panel_lines.append("")
                panel_lines.append(f"Live prediction:")
                panel_lines.append(f"{pred} ({conf:.0%})")

        panel_lines.extend(
            [
                "",
                "Score You/System/Tie:",
                f"{scores['user']} / {scores['system']} / {scores['tie']}",
            ]
        )

        display = make_display(frame)
        draw_panel_lines(
            display,
            frame.shape[1],
            panel_lines,
            start_y=35,
            line_gap=27,
            scale=0.55,
            thickness=2,
        )

        cv2.imshow("RPS Self Learning Camera", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("h"):
            print_help()

        if key in KEY_TO_LABEL:
            if features is None:
                last_message = "No hand found. Show your hand first."
                print("No hand was detected. Show your hand in front of the camera first.")
                continue

            label = KEY_TO_LABEL[key]
            append_sample(features, label)
            counts = dataset_counts()
            last_message = f"Saved sample as {label}. Counts: {counts}"
            print(f"Sample saved: {EN_LABELS[label]} | counts = {counts}")

        if key == ord("t"):
            ok_train, msg = train_model()
            print(msg)
            last_message = msg
            if ok_train:
                model = load_model()

        if key == ord("g"):
            if model is None:
                last_message = "No model. Collect samples then press T."
                print("No model exists yet. First collect samples for each class, then press t.")
                continue

            should_continue = play_round(cap, extractor, model, scores)
            model = load_model()
            if not should_continue:
                break

    cap.release()
    cv2.destroyAllWindows()

    if pygame is not None:
        try:
            pygame.mixer.quit()
        except Exception:
            pass

    print("\nProgram closed.")
    print(f"Final result => You: {scores['user']} | System: {scores['system']} | Tie: {scores['tie']}")


if __name__ == "__main__":
    main()