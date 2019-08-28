import numpy as np
import cv2
from matplotlib import pyplot as plt
from keypoint_matching import brute_force_matcher
import copy
import pygame
from pygame.locals import *
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from skimage.feature import match_descriptors
from helpers import essential_matrix_decomposition, R_t_from_P

class Extractor():
    def __init__(self,f,tx,ty,k):
        self.last = None
        self.K = k
        print(k)

    def extract(self, img):
        # Initiate STAR detector
        orb = cv2.ORB_create()
        #detection
        pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        if self.last is not None:
            # #brute force matching
            matches, cor = brute_force_matcher(self.last['des'], des, self.last['kp'], kp, 0.75)

            # print('cor:',cor)

            #essential should use normalised coordinates i.e. x_norm = K_inv @ x
            data = (cor[:, 0], cor[:, 1])
            model, inliers = ransac(data,
                              # EssentialMatrixTransform,
                              FundamentalMatrixTransform,
                              min_samples=8,
                              residual_threshold=0.3,
                              max_trials=100)
            matches = np.array(matches)
            #only consider inliers
            matches = matches[inliers]
            cor = cor[inliers]

            #get camera pose
            F = model.params
            E = self.K.T @ F @ self.K
            P1, P2 = essential_matrix_decomposition(E, cor)
            R, t = R_t_from_P(P2)
            print('R',R)
            print('t',t)
            # X_homos = cv2.triangulatePoints(P1,P2,x1s.T,x2s.T).T
            # Xs = [X_homo[:3] / X_homo[-1] for X_homo in X_homos]


            #denormalise?
            img_matches = drawMatches(img, self.last['kp'] ,kp, matches)
        else:
            matches = None
            img_matches = None

        self.last = {'kp' : kp, 'des' : des}
        return matches, img_matches


def drawMatches(image,kp1,kp2,matches,color=(255,0,0),line_color=(0,255,0)):
    img = copy.copy(image)
    for m in matches:

        x1 = round(kp1[m.queryIdx].pt[0])
        y1 = round(kp1[m.queryIdx].pt[1])

        x2 = round(kp2[m.trainIdx].pt[0])
        y2 = round(kp2[m.trainIdx].pt[1])

        cv2.circle(img,(x1, y1), 1, color, -1)
        cv2.circle(img,(x2, y2), 1, color, -1)
        cv2.line(img,(x1, y1),(x2, y2), line_color, 1)
    return img

def drawKeypoints(image,keypoints,color=(255,0,0)):
    img = copy.copy(image)
    for kp in keypoints:
        coord = kp.pt
        row = round(coord[0])
        col = round(coord[1])
        cv2.circle(img,(row, col), 1, color, -1)
    return img

import scipy.io
dispWidth = 960
dispHeight = 540
mat = scipy.io.loadmat('camera_calibration.mat')
k = mat['im']
fe = Extractor(1, dispWidth//2, dispHeight//2, k)

def process_frame(frame):
    matches, dispImg = fe.extract(frame)
    return dispImg


if __name__ == "__main__":
    cap = cv2.VideoCapture('video/test_countryroad.mp4')
    pygame.init()
    screen = pygame.display.set_mode((dispWidth, dispHeight))
    done = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame,(dispWidth,dispHeight))
            dispImg = process_frame(frame)
            if dispImg is not None:
                frame = cv2.cvtColor(dispImg, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)
                frame = np.flipud(frame)
                frame = pygame.surfarray.make_surface(frame)
                screen.blit(frame, (0, 0))
                pygame.display.update()
            for event in pygame.event.get():
                if event.type == QUIT:
                    done = True
            if done:
                break
        else:
            break
