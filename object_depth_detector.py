import numpy as np

class Object_Depth_Detector():
    def __init__(self):
        pass

    def __str__(self):
        return 'class/object description'

    def calculate_depth(self, depth_slice):
        """Calculates the trimmed minimum (closest distance within 2 standard deviations of the mean) depth within a
        given slice."""
        depths = depth_slice.flatten()
        depths = depths[depths>0]
        if depths.size == 0:
            return None
        else:
            mean = np.mean(depths)
            standard_deviation = np.std(depths)
            upper_bound = mean + 2 * standard_deviation
            lower_bound = mean - 2 * standard_deviation
            trimmed_depths = depths[np.where(np.logical_and(depths > lower_bound, depths < upper_bound))]
            if trimmed_depths.size == 0:
                return None
            else:
                trimmed_minimum = np.min(trimmed_depths)
                return trimmed_minimum

    def get_depths(self, boxes, depth_map):
        """ Input locations (bounding boxes) and depth map. Returns objects depths. Currently implemented for a single image."""
        depths = []#not efficient using append here - change
        [im_height, im_width] = depth_map.shape
        for box in boxes:
            [ymin, xmin, ymax, xmax] = box
            [left, right, top, bottom] = np.array([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]).astype(int)
            if(all([right>left, bottom>top])):
                depth = self.calculate_depth(depth_map[top:bottom,left:right])
                depths.append(depth)
        return depths

    def get_depth(self, box, depth_map):
        """ Input locations (bounding boxes) and depth map. Returns objects depths. Currently implemented for a single image."""
        depth = None
        [im_height, im_width] = depth_map.shape
        [ymin, xmin, ymax, xmax] = box
        [left, right, top, bottom] = np.array([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]).astype(int)
        if(all([right>left, bottom>top])):
            depth = self.calculate_depth(depth_map[top:bottom,left:right])
        return depth
