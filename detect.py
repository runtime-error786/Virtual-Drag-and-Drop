import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.9, maxHands=2)

rectangles = [
    [100, 100, 200, 200],
    [400, 100, 200, 200],
    [700, 100, 200, 200],
    [1000, 100, 200, 200]
]

selected_rectangle = -1
offset_x = 0
offset_y = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand = hands[0]
        lmList = hand['lmList']  
        bbox = hand['bbox']  

        fingers = detector.fingersUp(hand)
        if fingers[1] and fingers[2]:
            cursor = lmList[8]  
            cx, cy = cursor[0], cursor[1]

            if selected_rectangle == -1:
                for i, (x, y, w, h) in enumerate(rectangles):
                    if x < cx < x + w and y < cy < y + h:
                        selected_rectangle = i
                        offset_x = x - cx
                        offset_y = y - cy
                        break
            else:
                rectangles[selected_rectangle][0] = cx + offset_x
                rectangles[selected_rectangle][1] = cy + offset_y

        else:
            selected_rectangle = -1

    for x, y, w, h in rectangles:
        color = (0, 255, 0) if selected_rectangle != -1 else (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
