import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import joblib
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LABELS = ["rock", "paper", "scissors"]

FA_LABELS = {
    "rock": "سنگ / مشت",
    "paper": "کاغذ",
    "scissors": "قیچی",
}

DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "rps_dataset.csv"
MODEL_PATH = DATA_DIR / "rps_model.joblib"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    """
    تبدیل 21 landmark دست به یک بردار ویژگی ثابت.
    برای اینکه مدل به فاصله دست از دوربین حساس نباشد:
    - مختصات مچ دست را از همه نقاط کم می‌کنیم.
    - سپس همه چیز را نسبت به بیشترین فاصله نرمال می‌کنیم.
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    wrist = points[0].copy()
    points = points - wrist

    scale = np.max(np.linalg.norm(points, axis=1))
    if scale < 1e-6:
        scale = 1.0

    points = points / scale
    return points.flatten()


class HandFeatureExtractor:
    """
    تشخیص دست با MediaPipe و خروجی دادن feature برای مدل.
    """

    def __init__(self, max_num_hands: int = 1):
        self.mp_hands = mp_hands
        self.mp_draw = mp_drawing

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def extract(self, frame) -> Tuple[Optional[np.ndarray], Optional[object]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, None

        hand_landmarks = result.multi_hand_landmarks[0]
        features = normalize_landmarks(hand_landmarks)
        return features, hand_landmarks

    def draw(self, frame, hand_landmarks) -> None:
        if hand_landmarks is not None:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )


def append_sample(features: np.ndarray, label: str) -> None:
    """
    افزودن یک نمونه جدید به دیتاست محلی.
    """
    ensure_data_dir()

    is_new_file = not DATASET_PATH.exists()
    feature_names = [f"f{i}" for i in range(len(features))]

    with DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["label"] + feature_names)

        writer.writerow([label] + list(features.astype(float)))


def load_dataset() -> Optional[pd.DataFrame]:
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)
    if df.empty or "label" not in df.columns:
        return None

    return df


def dataset_counts() -> Dict[str, int]:
    df = load_dataset()
    if df is None:
        return {label: 0 for label in LABELS}

    counts = df["label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def can_train(min_per_class: int = 15) -> Tuple[bool, str]:
    counts = dataset_counts()
    missing = [f"{FA_LABELS[label]}: {counts[label]}/{min_per_class}" for label in LABELS if counts[label] < min_per_class]

    if missing:
        return False, "نمونه کافی نیست: " + " | ".join(missing)

    return True, "آماده آموزش است."


def train_model(test_size: float = 0.2, random_state: int = 42) -> Tuple[bool, str]:
    """
    آموزش مدل و ذخیره روی دیسک.
    """
    ensure_data_dir()

    df = load_dataset()
    if df is None:
        return False, "دیتاستی وجود ندارد."

    ok, msg = can_train(min_per_class=15)
    if not ok:
        return False, msg

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    return True, f"مدل آموزش داده شد و ذخیره شد. Accuracy تقریبی: {acc:.2%}"


def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def predict_move(model, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """
    خروجی:
    - label پیش‌بینی شده
    - confidence
    - probability هر کلاس
    """
    probs = model.predict_proba([features])[0]
    classes = list(model.classes_)

    prob_map = {label: 0.0 for label in LABELS}
    for cls, prob in zip(classes, probs):
        prob_map[cls] = float(prob)

    best_label = max(prob_map, key=prob_map.get)
    confidence = prob_map[best_label]
    return best_label, confidence, prob_map
