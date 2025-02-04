# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import gin
import numpy as np

from modules import holistic, translator

gin.parse_config_file('configs/holistic.gin')
gin.parse_config_file('configs/translator_inference.gin')
gin.parse_config_file('configs/utils.gin')


import collections # Import collections module

class Pipeline:

    def __init__(self):
        super().__init__()
        self.is_recording = True
        self.knn_records = []
        self.holistic_manager = holistic.HolisticManager()
        self.translator_manager = translator.TranslatorManager()

        self.frame_count = 0 # Initialize frame counter
        self.skip_frame = 1  # Initial skip frame rate
        self.motion_history = collections.deque(maxlen=5) # Keep a short history of motion

        self.reset_pipeline()

    def reset_pipeline(self):
        self.pose_history = []
        self.face_history = []
        self.lh_history = []
        self.rh_history = []

def update(self, frame_rgb):
    self.frame_count += 1
    if self.frame_count % self.skip_frame != 0:
        return  # Skip this frame

    h, w, _ = frame_rgb.shape
    assert h == w

    frame_res = self.holistic_manager(frame_rgb)

    if np.all(frame_res["pose_4d"] == 0.):
        return

    # Calculate motion stability (example: average hand keypoint velocity)
    current_hand_kps = np.concatenate([frame_res['lh_3d'].flatten(), frame_res['rh_3d'].flatten()])
    if self.motion_history:
        prev_hand_kps = self.motion_history[-1]
        motion_magnitude = np.linalg.norm(current_hand_kps - prev_hand_kps)
    else:
        motion_magnitude = 0
    self.motion_history.append(current_hand_kps)

    motion_threshold_low = 100.0  # Tunable thresholds - adjust as needed
    motion_threshold_high = 300.0 # Tunable thresholds - adjust as needed
    if motion_magnitude < motion_threshold_low:
        self.skip_frame = min(self.skip_frame + 1, 5)  # Increase skip (max skip 5, adjust as needed)
        cv2.putText(frame_rgb, "Stable, Skipping frames", (10, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1) # Debugging text
    elif motion_magnitude > motion_threshold_high:
        self.skip_frame = max(self.skip_frame - 1, 1)  # Decrease skip (min skip 1)
        cv2.putText(frame_rgb, "Motion Detected, Full FPS", (10, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1) # Debugging text
    else:
        self.skip_frame = 1 # Normal FPS when motion is in between thresholds
        cv2.putText(frame_rgb, "Normal FPS", (10, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1) # Debugging text


    if self.is_recording:
        cv2.putText(frame_rgb, "Recording...", (10, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 1)

        self.pose_history.append(frame_res["pose_4d"])
        self.face_history.append(frame_res["face_3d"])
        self.lh_history.append(frame_res["lh_3d"])
        self.rh_history.append(frame_res["rh_3d"])
