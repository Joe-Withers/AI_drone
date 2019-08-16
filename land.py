from djitellopy import Tello
import cv2
import time

tello = Tello()

tello.connect()
print('Battery',tello.get_battery())
tello.land()
tello.end()
