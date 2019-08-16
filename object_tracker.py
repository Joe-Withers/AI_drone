import numpy as np
import cv2

class Object_Tracker():

    def __init__(self, class_id=1, min_score=.65, max_frames_since_last_spotted=3):
        self.last_spotted = None
        self.class_id = class_id
        self.min_score = min_score
        self.last_patch = None
        self.last_box = None
        self.frames_since_last_spotted = 0
        self.max_frames_since_last_spotted = max_frames_since_last_spotted

    def prune_output_dict(self, output_dict):
        image_boxes = output_dict["detection_boxes"][np.logical_and(output_dict["detection_scores"]>self.min_score, output_dict["detection_classes"]==self.class_id)]
        classes = output_dict["detection_classes"][np.logical_and(output_dict["detection_scores"]>self.min_score, output_dict["detection_classes"]==self.class_id)]
        scores = output_dict["detection_scores"][np.logical_and(output_dict["detection_scores"]>self.min_score, output_dict["detection_classes"]==self.class_id)]
        return image_boxes, classes, scores

    def get_patch(self, current_frame, box):
        [im_height, im_width] = current_frame.shape #inefficient doing this inside the loop
        [ymin, xmin, ymax, xmax] = box
        [left, right, top, bottom] = np.array([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]).astype(int)
        return current_frame[ymin]

    def _sum_square_difference(self, im1, im2):
        im1 = np.array(im1)
        [y,x,_] = im1.shape
        im2 = np.array(cv2.resize(im2, (y, x)))
        ssd = np.sum((im2-im1)**2)
        return ssd

    def _euclidean_distance(self, box1, box2):
        [ymin1, xmin1, ymax1, xmax1] = box1
        center1 = np.array([(ymax1+ymin1)/2, (xmax1+xmin1)/2])

        [ymin2, xmin2, ymax2, xmax2] = box2
        center2 = np.array([(ymax2+ymin2)/2, (xmax2+xmin2)/2])

        return np.sqrt(np.sum((center1 + center2)**2))

    def patch_matching(self, current_frame, boxes):
        best_ssd = None
        best_idx = 0
        if self.last_patch is not None:
            for i, box in enumerate(boxes):
                patch = self.get_patch(current_frame, box)
                ssd = self._sum_square_difference(self.last_patch, patch)
                if (best_ssd is None) or (ssd < best_ssd):
                    best_ssd = ssd
                    best_idx = i
        self.last_patch = self.get_patch(current_frame, boxes[best_idx])
        self.last_box = boxes[best_idx]
        return best_idx

    def centroid_matching(self, boxes):
        best_euclid_dist = None
        best_idx = 0
        if self.last_box is not None:
            for i, box in enumerate(boxes):
                euclid_dist = self._euclidean_distance(self.last_box, box)
                if (best_euclid_dist is None) or (euclid_dist < best_euclid_dist):
                    best_euclid_dist = euclid_dist
                    best_idx = i
        self.last_box = boxes[best_idx]
        return best_idx

    def track_object(self, current_frame, output_dict):
        boxes, _, _ = self.prune_output_dict(output_dict)
        if len(boxes)>0:
            self.frames_since_last_spotted = 0
            idx = self.centroid_matching(boxes)
            return boxes[idx]
        else:
            self.frames_since_last_spotted += 1
            if self.frames_since_last_spotted > self.max_frames_since_last_spotted:
                self.last_box = None
            return self.last_box
