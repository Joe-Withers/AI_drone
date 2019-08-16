import numpy as np

class Planner():
    def __init__(self):
        self.depth_queue = Queue(5)
        self.box_height_queue = Queue(3)
        self.box_width_queue = Queue(3)

    def follow_object(self, box, depth, bounding_box, speed):
        commands = []
        if box is not None:
            [ymin, xmin, ymax, xmax] = box
            center = [(ymax+ymin)/2, (xmax+xmin)/2]
            commands.append(self.get_yaw(box, bounding_box, speed))
            commands.append(self.get_up_down(center, bounding_box, speed))
            commands.append(self.get_forward_back_box(box, bounding_box, speed))
        else:
            commands.append(('CLOCKWISE', 0))#explore mode - todo: make smarter
            commands.append(('UP', 0))
            commands.append(('FORWARD', 0))
        return commands

    def get_yaw(self, box, bounding_box, speed):
        [_, min_x, _, max_x, _, _] = bounding_box
        [ymin, xmin, ymax, xmax] = box
        center = [(ymax+ymin)/2, (xmax+xmin)/2]
        width = xmax - xmin
        height = ymax - ymin

        if center[1] < min_x or (xmax > 0.9 and width < 0.8):
            return 'CLOCKWISE', speed
        elif center[1] > max_x or (xmin < 0.1 and width < 0.8):
            return 'ANTICLOCKWISE', speed
        else:
            return 'CLOCKWISE', 0

    def get_up_down(self, center, bounding_box, speed):
        [min_y, _, max_y, _, _, _] = bounding_box

        if center[0] < min_y:
            return 'UP', 20
        elif center[0] > max_y:
            return 'DOWN', 20
        else:
            return 'UP', 0

    def get_forward_back_box(self, box, bounding_box, speed):
        [ymin, xmin, ymax, xmax] = box

        width = xmax - xmin
        # self.box_width_queue.addtoq(width)
        # avg_width = self.box_width_queue.get_mean()
        avg_width = width

        height = ymax - ymin
        # self.box_height_queue.addtoq(height)
        # avg_height = self.box_height_queue.get_mean()
        avg_height = height

        [_, _, _, _, min_size, max_size] = bounding_box

        if avg_width > max_size and avg_height > max_size:#0.2
            return 'BACK', speed
        elif avg_width < min_size or avg_height < min_size:#0.1
            return 'FORWARD', speed
        else:
            return 'FORWARD', 0


    def get_forward_back_depth(self, depth):
        self.depth_queue.addtoq(depth)
        avg_depth = self.depth_queue.get_mean()
        print('avg_depth',avg_depth)
        if avg_depth < 0.0013:
            return 'BACK', 20
        elif avg_depth > 0.0018:
            return 'FORWARD', 20
        else:
            return 'FORWARD', 0

class Queue():
    def __init__(self, max_vals):
      self.queue = list()
      self.max_vals = max_vals

    def addtoq(self,dataval):
    # Insert method to add element
      self.queue.insert(0,dataval)
      if len(self.queue) > self.max_vals:
          self.removefromq()

    # Pop method to remove element
    def removefromq(self):
      if len(self.queue)>0:
          return self.queue.pop()
      return ("No elements in Queue!")

    def get_mean(self):
        return sum(self.queue)/len(self.queue)
