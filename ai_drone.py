from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
from object_detector import Object_Detector
from depth_detector import Depth_Detector
from planner import Planner
from object_tracker import Object_Tracker
from object_depth_detector import Object_Depth_Detector


# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 20

def concat_images(im1,im2):
    im1 = cv2.resize(im1, (480, 360))
    im2 = cv2.resize(im2, (480, 360))
    return np.concatenate((im1, im2), axis=1)



class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
            - P: enter/exit follow person mode.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 360])


        # create object detector
        coco_frozen_graph_path = './ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
        coco_labels_path = './ssd_mobilenet_v2_coco_2018_03_29/mscoco_label_map.pbtxt'
        self.coco_obj_detector = Object_Detector(coco_frozen_graph_path, coco_labels_path, debug_mode = True)

        # create object detector
        face_frozen_graph_path = './ssd_face_detection/frozen_inference_graph_face.pb'
        face_labels_path = './ssd_face_detection/face_label_map.pbtxt'
        self.face_obj_detector = Object_Detector(face_frozen_graph_path, face_labels_path, debug_mode = True)
        #create tracker object
        self.tracker = Object_Tracker(class_id=1)
        # create depth detector
        self.depth_detector = Depth_Detector()
        self.object_depth_detector = Object_Depth_Detector()
        #create planner
        self.planner = Planner()

        self.face_bounding_box = [0.3, 0.4, 0.5, 0.6, 0.1, 0.2] #[min_y, min_x, max_y, max_x, min_size, max_size]
        self.person_bounding_box = [0.45, 0.45, 0.55, 0.55, 0.3, 0.5] #[min_y, min_x, max_y, max_x, min_size, max_size]

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.follow_human = False

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        should_stop = False
        while not should_stop:
            print('Battery: '+str(self.tello.get_battery())+'%')
            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                frame_read.stop()
                break

            self.screen.fill([0, 0, 0])
            current_frame = frame_read.frame

            #get depth map
            # depth_map, depth_img = self.depth_detector.detect_depth(current_frame)
            #detect objects
            output_dict, img = self.face_obj_detector.detect_objects(current_frame)
            #track object
            box = self.tracker.track_object(current_frame, output_dict)
            bounding_box = self.face_bounding_box
            # if box is None:#not found a face for n frames (where n is defined in tracker constructor)
            #     #so use coco object detector instead to look for a person
            #     #detect objects
            #     output_dict, img = self.coco_obj_detector.detect_objects(current_frame)
            #     #track object
            #     box = self.tracker.track_object(current_frame, output_dict)
            #     bounding_box = self.person_bounding_box
            #get objects depth
            # depth = self.object_depth_detector.get_depth(box, depth_map)
            depth = None # temp

            to_show = concat_images(img, current_frame)

            commands = self.planner.follow_object(box, depth, bounding_box, 60)
            if self.follow_human:
                for c in commands:
                    key, speed = c
                    self.make_move(key, speed)

            frame = cv2.cvtColor(to_show, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            # time.sleep(1 / FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def make_move(self, key, speed):
        if key == 'FORWARD':  # set forward velocity
            self.for_back_velocity = speed
        elif key == 'BACK':  # set backward velocity
            self.for_back_velocity = -speed
        elif key == 'LEFT':  # set left velocity
            self.left_right_velocity = -speed
        elif key == 'RIGHT':  # set right velocity
            self.left_right_velocity = speed
        elif key == 'UP':  # set up velocity
            self.up_down_velocity = speed
        elif key == 'DOWN':  # set down velocity
            self.up_down_velocity = -speed
        elif key == 'CLOCKWISE':  # set yaw clockwise velocity
            self.yaw_velocity = -speed
        elif key == 'ANTICLOCKWISE':  # set yaw counter clockwise velocity
            self.yaw_velocity = speed

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_p:
            self.follow_human = not self.follow_human

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)


def main():
    frontend = FrontEnd()
    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
