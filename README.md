# VisionAssist (SASEHACK26)
A wearable, hands-free AI navigation assistant that utilizes edge-based computer vision and spatial audio to help visually impaired individuals detect hazards and navigate physical environments.

# Visionary: Wearable AI Navigation Assistant
*Developed for SASEhack 2026 by an Electrical Engineering student at Texas A&M University*

## 1. Inspiration & Real-World Impact
Navigating dynamic physical environments and completing daily tasks is a profound challenge for visually impaired individuals. While white canes detect ground-level barriers, they frequently miss head-height hazards and cannot identify specific objects. This device acts as a multimodal, hands-free wearable that provides real-time spatial awareness and task guidance. By explicitly targeting the **Best Education, Accessibility, or Social Impact** track, this prototype solves a real problem for real people by translating visual data into spatial audio cues, restoring independence without relying on expensive, proprietary hardware.

## 2. Hardware Architecture
Submitted for the **Best Hardware Hack** track, the physical prototype is designed to be affordable, low-latency, and highly wearable.
* **Vision Processing:** A Raspberry Pi 4 runs the inference logic locally, capturing real-time environmental footage through a Sony IMX323 USB camera mounted at chest level.
* **Adaptive Lighting:** Two 5W/3W 6000K LEDs powered by dual 3W LED drivers provide steady illumination in low-light environments, controlled dynamically via Pi GPIOs to prevent image overexposure.
* **Haptics & Safety Controller:** An ESP32 or Arduino Nano serves as a deterministic safety controller. It drives a passive buzzer to deliver immediate proximity beeps when obstacles enter a critical radius, triggered alongside a physical push-button module for mode switching.
* **Altimetry:** A GY-MS5837 pressure sensor acts as a high-precision barometric altimeter to detect elevation shifts, such as approaching staircases or uneven terrain.

## 3. Software Stack & Execution
The software prioritizes edge processing to ensure reliable functionality even without internet access. This technical explanation is provided so judges can successfully verify the execution and run the code directly.
* **Obstacle Tracking:** A lightweight quantized `YOLOv8` model runs via Ultralytics and OpenCV to segment paths and track incoming objects at a high frame rate.
* **Audio Feedback:** The system utilizes `pyttsx3` with local ALSA/espeak drivers to synthesize real-time voice instructions managed by an asynchronous threading queue.
* **Proximity Logic:** Custom bounding-box ratios calculate the relative distance and spatial position (Left, Center, Right) of classified hazards.

**How to run locally:**
1. Clone this repository to a Raspberry Pi 4 running a PEP-668 compliant virtual environment.
2. Install system audio dependencies via the terminal: `sudo apt install espeak-ng libespeak1 ffmpeg libasound2-dev`.
3. Activate the environment and install Python libraries: `pip install opencv-python ultralytics pyttsx3`.
4. Connect the Sony IMX323 via USB and the Arduino microcontroller via the serial port.
5. Execute the main tracking loop: `python3 main.py`.
