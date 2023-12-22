import numpy as np
import cv2
import mediapipe as mp
import time
import HandTrackingModule as HTM
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam_width, cam_height = 640, 480
cTime, pTime= 0, 0
hand_detector = HTM.HandDetector(min_detection_confidence=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

while True:
    success, img = cap.read()
    img = hand_detector.findHands(img)
    lm_list = hand_detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        #print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        new_vol = np.interp(length, [50, 300], [min_vol, max_vol])
        bar_vol = np.interp(length, [50, 300], [400, 150])
        per_vol = np.interp(length, [50, 300], [0, 100])
        volume.SetMasterVolumeLevel(new_vol, None)
        if length < 50:
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(bar_vol)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per_vol)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break