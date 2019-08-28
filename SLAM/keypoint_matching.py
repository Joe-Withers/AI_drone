import numpy as np
import numpy.matlib
import cv2

def brute_force_matcher(des1, des2, kp1, kp2, r):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    ret = []
    mat = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()

    for m,n in matches:
        if m.distance < r*n.distance:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt

            # be within orb distance 32
            if m.distance < 32:
                # keep around indices
                # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    mat.append(m)
                    ret.append((p1, p2))
    return mat, np.array(ret)

def brute_force_matcher_mine(features1, features2, kps1, kps2, r):
    features1 = np.array(features1)
    features2 = np.array(features2)
    coords1 = np.array([kp.pt for kp in kps1])
    coords2 = np.array([kp.pt for kp in kps2])


    [num_features1, _] = features1.shape
    [num_features2, _] = features2.shape
    # each match contains [index of feature 1, index of feature 2, error]
    sorted_err = np.zeros((num_features1, num_features2))
    sorted_idx = np.zeros((num_features1, num_features2))

    #for each feature in features1, order features2 based on error
    for i in range(num_features1):
        err = np.sum(np.abs(features2 - np.matlib.repmat(features1[i,:],num_features2,1)),axis=1)
        sorted_err[i,:] = np.sort(err)
        sorted_idx[i,:] = np.argsort(err)

    # find rows which are not mutally the best
    rows_to_ignore = []
    for i in range(num_features2):
        if(len(np.where(sorted_idx[:,0]==i)) > 1):
            errs = sorted_err[sorted_idx[:,0]==i,0]
            indexes = np.where(sorted_idx[:,0]==i)
            rows_to_ignore = [rows_to_ignore, indexes[errs >  np.min(errs)].T]

    # ratio testing
    matches = []
    for i in range(num_features1):
        if(i not in rows_to_ignore):
            err_ratio = sorted_err[i,0]/sorted_err[i,1]
            if(err_ratio < r):
                # matches.append([i,sorted_idx[i,0]])#rewrite without append for more speed
                matches.append(cv2.DMatch(i, int(sorted_idx[i,0]), sorted_err[i,0]))

    # matches = np.array(matches).astype(int)
    # correspondences = np.array([[coords1[match[0],:], coords2[match[1],:]] for match in matches])
    correspondences = np.array([(coords1[match.queryIdx,:],coords2[match.trainIdx,:]) for match in matches])

    return matches, correspondences
