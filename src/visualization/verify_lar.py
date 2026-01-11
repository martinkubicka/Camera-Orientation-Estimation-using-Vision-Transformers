import cv2 
import lib.Perspec2Equirec as P2E
import numpy as np

# deg all
fov = 59.030696
pitch = 3.486826 # -pi;pi 
yaw = 43.815954 # -pi;pi 0 at center and pi at right
roll = -2.938603 # -pi;pi 0 aligned pi/2 -> 90deg to right (clockwise)

equ = P2E.Perspective("./data/8/photo.jpg", FOV=fov, THETA=yaw, PHI=pitch, ROLL=roll)
img, mask = equ.GetEquirec(height=2072, width=4144)
cv2.imwrite("output_with_roll.png", img)
    