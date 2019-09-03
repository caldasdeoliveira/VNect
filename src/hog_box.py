#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from __future__ import print_function
import cv2
import numpy as np

from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import pdb

class HOGBox:
    """
    a simple HOG-method-based human tracking box
    """
    # mouse click flag
    clicked = False

    def __init__(self):
        print('Initializing HOGBox...')
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._box_init_window_name = 'Bounding Box Initialization'
        cv2.namedWindow(self._box_init_window_name)
        cv2.setMouseCallback(self._box_init_window_name, self.on_mouse)
        print('HOGBox initialized.')

    def __call__(self, img, original=True):
        H, W = img.shape[:2]
        if original:
            found, w = self.hog.detectMultiScale(img)
            rect = self.cal_rect(found[np.argmax([found[i, 2] * found[i, 3] for i in range(len(found))])], H, W) \
                if len(found) else [0, 0, W, H]  # biggest area
            # rect = self.cal_rect(found[np.argmax(w)], H, W) if len(found) else [0, 0, W, H]  # biggest weight
            self.draw_rect(img, rect)
            scale = 400 / H
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            cv2.imshow(self._box_init_window_name, img)

            if self.clicked:
                cv2.destroyWindow(self._box_init_window_name)
            return self.clicked, rect
        else:
            min_width=800
            image = imutils.resize(img, width=min(min_width, img.shape[1]))
            # detect people in the image
            (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.01)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            rect=[]
            
            for p in pick:
                rect = [p[0],p[1],p[2]-p[0],p[3]-p[1]]
                rect =[x * img.shape[1] // min(min_width, img.shape[1]) for x in rect]
                # rect = self.cal_rect(found[np.argmax([found[i, 2] * found[i, 3] for i in range(len(found))])], H, W) \
                # if len(found) else [0, 0, W, H]  # biggest area
                # rect = self.cal_rect(found[np.argmax(w)], H, W) if len(found) else [0, 0, W, H]  # biggest weight
                self.draw_rect(img, rect)
            scale = 400 / H
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            cv2.imshow(self._box_init_window_name, img)

            if self.clicked:
                cv2.destroyWindow(self._box_init_window_name)
            return self.clicked, rect


    def on_mouse(self, event, x, y, flags, param):
        """
        attain mouse clicking message
        """
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = True

    @staticmethod
    def cal_rect(rect, H, W):
        """
        calculate the box size and position
        """
        x, y, w, h = rect
        offset_w = int(0.4 / 2 * W)
        offset_h = int(0.2 / 2 * H)
        return [np.max([x - offset_w, 0]),  # x
                np.max([y - offset_h, 0]),  # y
                np.min([x + w + offset_w, W]) - np.max([x - offset_w, 0]),  # w
                np.min([y + h + offset_h, H]) - np.max([y - offset_h, 0])]  # h

    @staticmethod
    def draw_rect(img, rect):
        """
        draw bounding box in the BB initialization window, and record current rect (x, y, w, h)
        """
        x, y, w, h = rect
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, (28, 76, 242), 4)
