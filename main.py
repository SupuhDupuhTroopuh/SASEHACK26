import cv2
import pyttsx3
import time
import threading
import queue
from ultralytics import YOLO


# MODEL

model = YOLO("yolov8s.pt")


# CAMERA

# cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open USB camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# SPEECH THREAD

speech_queue = queue.Queue(maxsize=1)


def speech_worker():
    """
    Runs separately from the camera loop.
    """

    engine = pyttsx3.init()

    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)

    while True:

        message = speech_queue.get()

        if message is None:
            break

        print("SPEAK:", message)

        engine.say(message)
        engine.runAndWait()

        speech_queue.task_done()

    engine.stop()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak_object(object_name, proximity):

    if proximity == "VERY NEAR":
        message = (
            f"Warning. {object_name} ahead. "
            f"Very near."
        )

    else:
        message = f"{object_name} ahead."

    # Don't let speech messages pile up
    if speech_queue.empty():

        try:
            speech_queue.put_nowait(message)

        except queue.Full:
            pass


# PROXIMITY

REFERENCE_HEIGHTS = {
    "person": 300,
    "bicycle": 250,
    "car": 220,
    "motorcycle": 220,
    "bus": 300,
    "truck": 250,

    "chair": 180,
    "bench": 180,
    "backpack": 120,
    "suitcase": 160,

    "bottle": 100,
    "cup": 90,
    "wine glass": 80,
    "bowl": 100,

    "laptop": 100,
    "cell phone": 80,
    "book": 100,
    "keyboard": 100,

    "default": 150
}


def get_proximity(
    object_name,
    x1,
    y1,
    x2,
    y2,
    frame_height
):

    box_height = y2 - y1

    expected_height = REFERENCE_HEIGHTS.get(
        object_name,
        REFERENCE_HEIGHTS["default"]
    )

    size_ratio = box_height / expected_height

    bottom_ratio = y2 / frame_height

    proximity_score = (
        0.6 * size_ratio +
        0.4 * bottom_ratio
    )

    if proximity_score < 0.55:
        return "FAR"

    elif proximity_score < 1.0:
        return "NEAR"

    else:
        return "VERY NEAR"


# MAIN LOOP

CONFIDENCE = 0.35

frame_count = 0

DETECTION_INTERVAL = 2

last_results = None

last_spoken = {}

SPEECH_COOLDOWN = 4.0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Couldn't read frame")
        break

    frame_count += 1

    frame_height, frame_width = frame.shape[:2]


        # YOLO
    
    if frame_count % DETECTION_INTERVAL == 0:

        last_results = model.track(
            frame,
            persist=True,
            imgsz=512,
            conf=CONFIDENCE,
            verbose=False
        )


        # PROCESS DETECTIONS
    
    if last_results is not None:

        current_time = time.time()

        for result in last_results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                object_name = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                                # TRACK ID
                
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1


                                # POSITION
                
                center_x = (x1 + x2) // 2

                if center_x < frame_width / 3:

                    position = "LEFT"

                elif center_x < 2 * frame_width / 3:

                    position = "CENTER"

                else:

                    position = "RIGHT"


                                # PROXIMITY
                
                proximity = get_proximity(
                    object_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    frame_height
                )


                                # SPEECH
                
                if (
                    position == "CENTER"
                    and proximity in ["NEAR", "VERY NEAR"]
                    and track_id != -1
                ):

                    last_time = last_spoken.get(
                        track_id,
                        0
                    )

                    if (
                        current_time - last_time
                        >= SPEECH_COOLDOWN
                    ):

                        speak_object(
                            object_name,
                            proximity
                        )

                        last_spoken[track_id] = (
                            current_time
                        )


                                # DRAW
                
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{object_name} "
                    f"{confidence:.2f} "
                    f"ID:{track_id} "
                    f"{position} "
                    f"{proximity}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


        # DISPLAY
    
    cv2.imshow(
        "Obstacle Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# CLEANUP

cap.release()

cv2.destroyAllWindows()

speech_queue.put(None)

speech_thread.join(timeout=1)
