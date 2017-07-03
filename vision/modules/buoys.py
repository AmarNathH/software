#!/usr/bin/env python3

""" TODO
- Tune segmentation for green buoy (may need to use similar components as red,
  because TRANSDEC buoy and lighting is better)
- Detection of (yellow?) (propeller?) may need a separate vision module
  depending on its shape
- Buoy reference colors and segmentation thresholds will need to be tuned for
  TRANSDEC (See competition buoys on loglan)
"""

import math
import time

from collections import namedtuple
from functools import reduce
from itertools import permutations

import cv2
import numpy as np

import shm

from vision.modules.base import ModuleBase
from vision.vision_common import green, red, white, yellow
from vision import options
from color_balance import balance

options = [ options.BoolOption('debug', False),
            options.BoolOption('red_debug', False),
            options.BoolOption('green_debug', False),
            options.BoolOption('yellow_debug', False),
            options.BoolOption('use_color_correction', False),
            options.IntOption('hls_l_min', 0, 0, 255),
            options.IntOption('hls_l_max', 255, 0, 255),
            options.IntOption('red_hls_h_min', 0, 0, 255),
            options.IntOption('red_hls_h_max', 50, 0, 255),
            options.IntOption('red_ycrcb_cb_min', 0, 0, 255),
            options.IntOption('red_ycrcb_cb_max', 115, 0, 255),
            options.IntOption('red_lab_a_min', 130, 0, 255),
            options.IntOption('red_lab_a_max', 255, 0, 255),
            options.IntOption('red_lab_b_min', 145, 0, 255),
            options.IntOption('red_lab_b_max', 230, 0, 255),
            options.IntOption('red_erode_kernel_size', 1, 1, 151),
            options.IntOption('red_dilate_kernel_size', 1, 1, 151),
            options.IntOption('green_lab_l_min', 80, 0, 255),
            options.IntOption('green_lab_l_max', 141, 0, 255),
            options.IntOption('green_hsv_v_min', 29, 0, 255),
            options.IntOption('green_hsv_v_max', 147, 0, 255),
            options.IntOption('green_hsv_h_min', 28, 0, 255),
            options.IntOption('green_hsv_h_max', 66, 0, 255),
            options.IntOption('green_ycrcb_y_min', 16, 0, 255),
            options.IntOption('green_ycrcb_y_max', 121, 0, 255),
            options.IntOption('green_erode_kernel_size', 4, 1, 151),
            options.IntOption('green_dilate_kernel_size', 3, 1, 151),
            options.IntOption('yellow_hls_h_min', 17, 0, 255),
            options.IntOption('yellow_hls_h_max', 49, 0, 255),
            options.IntOption('yellow_ycrcb_cb_min', 0, 0, 255),
            options.IntOption('yellow_ycrcb_cb_max', 111, 0, 255),
            options.IntOption('yellow_lab_a_min', 0, 0, 255),
            options.IntOption('yellow_lab_a_max', 200, 0, 255),
            options.IntOption('yellow_lab_b_min', 154, 0, 255),
            options.IntOption('yellow_lab_b_max', 255, 0, 255),
            options.IntOption('yellow_erode_kernel_size', 1, 1, 151),
            options.IntOption('yellow_dilate_kernel_size', 1, 1, 151),
            options.IntOption('blur_kernel', 1, 1, 151),
            options.IntOption('max_error', 5000, 100, 100000),
            options.IntOption('max_buoy_error', 1500, 100, 100000),
            options.DoubleOption('min_circularity', 0.62, 0.1, 0.95),
            options.DoubleOption('min_percent_frame', 0.001, 0, 0.01),
            options.IntOption('ex', 243, 0, 1600),
            options.IntOption('ey', 171, 0, 1200),
            options.IntOption('max_y_diff', 50, 0, 500) ]

# L, A, B
GREEN_REF = (100, 100, 150)
YELLOW_REF = (180, 100, 180)
RED_REF = (150, 150, 160)

#RED_REF = (150, 180, 180)
#GREEN_REF = (110, 115, 150)
#YELLOW_REF = (225, 110, 200)
#RED_REF = (100, 170, 170)
#GREEN_REF = (100, 80, 170)
#YELLOW_REF = (140, 114, 180)

COLOR_ORDER = (green, yellow, red)
DOWNSCALE_RATIO = 0.33
CONTOUR_CIRCULARITY_HEURISTIC_LIMIT = 50
CONTOUR_SCALED_HEURISTIC_LIMIT = 20
CONTOUR_CLASSIFICATION_LIMIT = 5
ContourAreaData = namedtuple('ContourAreaData', ['contour', 'area'])
ContourScoreData = namedtuple('ContourScoreData', ['contour', 'area', 'circularity', 'score', 'center', 'radius'])

INVALID_BUOY_ERROR = 1e99


def tag(mat, text, pos):
    position = (pos[0]-100, pos[1])
    cv2.putText(mat, text, position, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), thickness=2)


class BuoyData:
    def __init__(self, shm_group, color, reference_color):
        self.shm_group = shm_group
        self.results = shm_group.get()
        self.color = color
        self.reference_color = reference_color
        self.reset_frame_state()

    def reset_frame_state(self):
        self.contour = None

    def zero_results(self):
        self.results.heuristic_score = 0
        self.results.probability = 0
        self.results.area = 0
        self.results.percent_frame = 0

    def set_results(self, image_width, image_height, contour_list):
        assert self.contour is not None # This must be called after contour has been set.
        self.results.center_x = int(1 / DOWNSCALE_RATIO) * int(self.contour.center[0])
        self.results.center_y = int(1 / DOWNSCALE_RATIO) * int(self.contour.center[1])
        self.results.top_x = int(1 / DOWNSCALE_RATIO) * int(self.contour.center[0])
        self.results.top_y = int(1 / DOWNSCALE_RATIO) * int(self.contour.center[1] - self.contour.radius / 2)
        self.results.r_side_x = int(1 / DOWNSCALE_RATIO) * int(self.contour.center[0] + self.contour.radius * 2)
        self.results.area = self.contour.area
        self.results.heuristic_score = self.contour.score
        self.results.percent_frame = 100 * self.contour.area / (image_width * image_height)
        #print("Percent: {}".format(str(self.results.percent_frame)))
        self.results.frame_width = image_width
        self.results.frame_height = image_height
        self.results.probability = self.contour.score / reduce(lambda acc, x: acc + x.score, contour_list, 0)
        # TODO XXX HAX Please fix this probabilities.
        self.results.probability = max(self.results.probability, 0.5)

    def set_shm_group(self):
        self.shm_group.set(self.results)

RED_BUOY = BuoyData(shm.red_buoy_results, red, RED_REF)
GREEN_BUOY = BuoyData(shm.green_buoy_results, green, GREEN_REF)
YELLOW_BUOY = BuoyData(shm.yellow_buoy_results, yellow, YELLOW_REF)
BUOYS = [RED_BUOY, GREEN_BUOY, YELLOW_BUOY]

class Buoys(ModuleBase):


    def calculate_error(self, buoy, candidate):
        if buoy is None:
            return INVALID_BUOY_ERROR
        error_vec = np.array(buoy.reference_color) - \
                    np.array(candidate[1])
        buoy_error = error_vec.dot(error_vec)
        if buoy_error > self.options['max_buoy_error']:
            return INVALID_BUOY_ERROR
        #if abs(buoy.results.center_y - 600) > self.options['max_dist_from_center']:
        #    return INVALID_BUOY_ERROR
        else:
            return buoy_error

    def add_ordered_center_x(self, buoy_list, buoy):
        list = []
        if buoy_list is None:
            list.append(buoy)
            return list
        list = buoy_list
        for i in range(len(list)):
            if buoy.results.center_x < list[i].results.center_x:
                list.insert(i, buoy)
                return list
        list.append(buoy)
        return list

    def compare_element_orders(self, list1, list2):
        k = 0
        for i in range(len(list1)):
            for j in range(len(list2)):
                if list1[i] == list2[j]:
                    if j < k:

                        return False
                    else:
                        k = j
        return True

    def check_pos(self, mapping, classification):
        #ordering = []
        y_coords = []
        #color_order = []
        for buoy in mapping:
            #if classification[buoy] is None and buoy is not None:
            if classification[buoy] is not None:
                #ordering = self.add_ordered_center_x(ordering, buoy)
                for y in y_coords:
                    #print("{}".format(classification[buoy].center[1] - y))
                    if abs(classification[buoy].center[1] - y) > self.options['max_y_diff']:
                            return False
                y_coords.append(classification[buoy].center[1])
        return True


    def classify_buoys(self, buoys, candidates):
        """
            buoys is a list of BuoyData objects
            candidates is a list of (candidate, color) tuples
        """
        # Possible mappings for the # of candidates to the 3 buoys
        buoys = list(buoys)
        while len(candidates) > len(buoys):
            buoys.append(None)
        possible_mappings = permutations(buoys, len(candidates))
        assignments = []
        max_buoys_classified = 0
        for mapping in possible_mappings:
            total_error = 0
            classification = {buoy: None for buoy in mapping if buoy is not None}
            buoys_classified = 0
            # Iterate through one-to-one mappings of buoys to candidates
            for i in range(len(candidates)):
                error = self.calculate_error(mapping[i], candidates[i])
                if error >= INVALID_BUOY_ERROR:
                    classification[mapping[i]] = None
                else:
                    classification[mapping[i]] = candidates[i][0]
                    #print("{}".format(classification[mapping[i]]))
                    total_error += error
                    buoys_classified += 1
            # UNCOMMENT THIS WHEN TUNING BUOY REFERENCE COLORS
            #print("Considering: " + str([buoy.color if\
            #        classification[buoy] is not None else (-1, -1, -1) for buoy in mapping]))
            #print(total_error)

            if buoys_classified > 0 and total_error < self.options['max_error'] and buoys_classified >= max_buoys_classified and self.check_pos(mapping, classification):
                max_buoys_classified = buoys_classified
                assignments.append((total_error, buoys_classified, classification))

        # Consider assignments with the most number of classified buoys
        assignments = filter(lambda x: x[1] == max_buoys_classified, assignments)
        min_error = self.options['max_error']
        optimal_assignment = (0, 0, {buoy: None for buoy in buoys})
        for assignment in assignments:
            if assignment[0] < min_error:
                min_error = assignment[0]
                optimal_assignment = assignment
        #print(len(optimal_assignment[2]), min_error)
        for (k, v) in optimal_assignment[2].items():
            if v is not None:
                for candidate in enumerate(candidates):
                    if candidate[1][0].center == v.center:
                        #print(k.color, candidate[1][1])
                        break
        return optimal_assignment

    def process(self, mat):
        start_time = time.time()
        
        mat = cv2.resize(mat, (int(mat.shape[1] * DOWNSCALE_RATIO), int(mat.shape[0] * DOWNSCALE_RATIO)))
        image_size = mat.shape[1] * mat.shape[0]

        if self.options['use_color_correction']:
            mat = balance(mat)

        self.post('orig', mat)
        blurred = cv2.GaussianBlur(mat,
                (self.options_dict['blur_kernel'].value * 2 + 1,
                 self.options_dict['blur_kernel'].value * 2 + 1),
                0)

        mask_e = np.zeros((mat.shape[0], mat.shape[1]), np.uint8)
        cv2.ellipse(mask_e,(int(mask_e.shape[1]/2), int(mask_e.shape[0]/2)),(self.options['ex'],self.options['ey']),0,0,360,(255,255,255),-1)

        if self.options['debug']:
            self.post('mask', mask_e)

        #self.post('balanced', mat)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        hls_split = cv2.split(hls)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        lab_split = cv2.split(lab)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        ycrcb_split = cv2.split(ycrcb)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv_split = cv2.split(hsv)

        red_a_threshed = cv2.inRange(lab_split[1], self.options['red_lab_a_min'],
                                                   self.options['red_lab_a_max'])
        if self.options['red_debug']:
            self.post('red_a_threshed', red_a_threshed)

        red_b_threshed = cv2.inRange(lab_split[2], self.options['red_lab_b_min'],
                                                   self.options['red_lab_b_max'])
        if self.options['red_debug']:
            self.post('red_b_threshed', red_b_threshed)

        red_cb_threshed = cv2.inRange(ycrcb_split[2], self.options['red_ycrcb_cb_min'],
                                                      self.options['red_ycrcb_cb_max'])
        if self.options['red_debug']:
            self.post('red_cb_threshed', red_cb_threshed)

        red_h_threshed = cv2.inRange(hls_split[0], self.options['red_hls_h_min'],
                                                   self.options['red_hls_h_max'])
        if self.options['red_debug']:
            self.post('red_h_threshed', red_h_threshed)

        red_threshed = red_a_threshed & red_b_threshed & red_h_threshed & red_cb_threshed & mask_e

        red_morphed = cv2.erode(red_threshed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['red_erode_kernel_size'],
                         self.options['red_erode_kernel_size'])))
        red_morphed = cv2.dilate(red_morphed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['red_dilate_kernel_size'],
                         self.options['red_dilate_kernel_size'])))

        if self.options['red_debug']:
            self.post('red_threshed', red_threshed)
            self.post('red_morphed', red_morphed)

        _, red_contours, _ = cv2.findContours(red_morphed.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        

        yellow_a_threshed = cv2.inRange(lab_split[1], self.options['yellow_lab_a_min'],
                                                   self.options['yellow_lab_a_max'])
        if self.options['yellow_debug']:
            self.post('yellow_a_threshed', yellow_a_threshed)

        yellow_b_threshed = cv2.inRange(lab_split[2], self.options['yellow_lab_b_min'],
                                                   self.options['yellow_lab_b_max'])
        if self.options['yellow_debug']:
            self.post('yellow_b_threshed', yellow_b_threshed)

        yellow_cb_threshed = cv2.inRange(ycrcb_split[2], self.options['yellow_ycrcb_cb_min'],
                                                      self.options['yellow_ycrcb_cb_max'])
        if self.options['yellow_debug']:
            self.post('yellow_cb_threshed', yellow_cb_threshed)

        yellow_h_threshed = cv2.inRange(hls_split[0], self.options['yellow_hls_h_min'],
                                                   self.options['yellow_hls_h_max'])
        if self.options['yellow_debug']:
            self.post('yellow_h_threshed', yellow_h_threshed)

        yellow_threshed = yellow_a_threshed & yellow_b_threshed & yellow_h_threshed & yellow_cb_threshed & mask_e

        yellow_morphed = cv2.erode(yellow_threshed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['yellow_erode_kernel_size'],
                         self.options['yellow_erode_kernel_size'])))
        yellow_morphed = cv2.dilate(yellow_morphed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['yellow_dilate_kernel_size'],
                         self.options['yellow_dilate_kernel_size'])))

        if self.options['yellow_debug']:
            self.post('yellow_threshed', yellow_threshed)
            self.post('yellow_morphed', yellow_morphed)

        _, yellow_contours, _ = cv2.findContours(yellow_morphed.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)


        green_l_threshed = cv2.inRange(lab_split[0], self.options['green_lab_l_min'],
                                                     self.options['green_lab_l_max'])
        if self.options['green_debug']:
            self.post('green_l_threshed', green_l_threshed)

        green_v_threshed = cv2.inRange(hsv_split[2], self.options['green_hsv_v_min'],
                                                     self.options['green_hsv_v_max'])
        if self.options['green_debug']:
            self.post('green_v_threshed', green_v_threshed)

        green_h_threshed = cv2.inRange(hsv_split[0], self.options['green_hsv_h_min'],
                                                     self.options['green_hsv_h_max'])
        if self.options['green_debug']:
            self.post('green_h_threshed', green_h_threshed)

        green_y_threshed = cv2.inRange(ycrcb_split[0], self.options['green_ycrcb_y_min'],
                                                       self.options['green_ycrcb_y_max'])
        if self.options['green_debug']:
            self.post('green_y_threshed', green_y_threshed)

        green_threshed = green_l_threshed & green_v_threshed & green_h_threshed & green_y_threshed & mask_e

        green_morphed = cv2.erode(green_threshed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['green_erode_kernel_size'],
                         self.options['green_erode_kernel_size'])))
        green_morphed = cv2.dilate(green_morphed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['green_dilate_kernel_size'],
                         self.options['green_dilate_kernel_size'])))

        green_morphed = green_morphed - yellow_morphed - red_morphed

        if self.options['green_debug']:
            self.post('green_threshed', green_threshed)
            self.post('green_morphed', green_morphed)

        _, green_contours, _ = cv2.findContours(green_morphed.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        contours = []
        contours.extend(red_contours)
        contours.extend(green_contours)
        contours.extend(yellow_contours)

        circles_mat = lab_split[2].copy()
        circles_mat = cv2.medianBlur(circles_mat, 11)
        circles = cv2.HoughCircles(circles_mat, cv2.HOUGH_GRADIENT, 1, 50,
                param1=100, param2=30, minRadius=1, maxRadius=200)
        if self.options['red_debug']:
            self.post("circles_mat", circles_mat)

        if circles is not None:
            circle_mask = np.zeros(mat.shape, np.uint8)
            for circle in circles[0, :]:
                #print(circle)
                cv2.circle(circle_mask, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
            if self.options['red_debug']:
                self.post("circle_mask", circle_mask)
            #self.post("img_mat", mat.copy() & circle_mask)
            morphed_hough_heuristic = red_morphed.copy() & cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)
            _, heuristic_contours, _ = cv2.findContours(morphed_hough_heuristic.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            #contours.extend(heuristic_contours)
            #self.post("morphed_hough_heuristic", morphed_hough_heuristic)


        contourAreas = []
        for contour in contours:
            contourArea = cv2.contourArea(contour)
            if contourArea >= image_size * self.options['min_percent_frame']:
                contourAreas.append(ContourAreaData(contour, contourArea))
        # Sort contours in descending order by their areas
        contourAreas = sorted(contourAreas, key=lambda x: -x.area)[:CONTOUR_CIRCULARITY_HEURISTIC_LIMIT]

        contourScores = []
        for contourArea in contourAreas:
            center, radius = cv2.minEnclosingCircle(contourArea.contour)
            circularity = contourArea.area / (math.pi * radius ** 2)
            if circularity >= self.options['min_circularity']:
                # Calculate a heuristic score defined by
                # (contour cirularity)^1.25 * (contour area)
                heuristic_score = (circularity**1.25) * contourArea.area
                contourScores.append(ContourScoreData(contourArea.contour,
                                     contourArea.area, circularity, heuristic_score, center, radius))
        # Sort contours in descending order by their heuristic score
        contourScores = sorted(contourScores, key=lambda x: -x.score)[:CONTOUR_SCALED_HEURISTIC_LIMIT]

        if contourScores:

            if self.options['debug']:
                contoursMat = mat.copy()
                cv2.drawContours(contoursMat, [cand.contour for cand in contourScores], -1, white, 3)
                self.post('contours', contoursMat)
                avgMat = contoursMat.copy()

            buoyContours = sorted(contourScores, key=lambda x: -x.score)[:CONTOUR_CLASSIFICATION_LIMIT]
            total_mask = np.zeros(mat.shape, np.uint8)
            buoy_results = []
            # Only look at top contour candidates.
            for i, buoy_candidate in enumerate(buoyContours):
                # Calculate average L, A, and B values for each contour
                mask = np.zeros(lab.shape, np.uint8)
                cv2.drawContours(mask, [buoy_candidate.contour], 0, white, cv2.FILLED)
                if i == 0:
                    cv2.drawContours(total_mask, [buoy_candidate.contour], 0, green, 1)
                else:
                    cv2.drawContours(total_mask, [buoy_candidate.contour], 0, white, 1)

                just_buoy = cv2.bitwise_and(lab, mask)
                just_buoy_split = cv2.split(just_buoy)

                l_ch = just_buoy_split[0]
                a_ch = just_buoy_split[1]
                b_ch = just_buoy_split[2]

                l_ch_nonzero = l_ch[l_ch != 0]
                a_ch_nonzero = a_ch[a_ch != 0]
                b_ch_nonzero = b_ch[b_ch != 0]

                # Clip outliers
                if len(l_ch) != 0:
                    l_min_max = np.percentile(l_ch_nonzero, [0.2, 99.8])
                a_min_max = np.percentile(a_ch_nonzero, [0.2, 99.8])
                b_min_max = np.percentile(b_ch_nonzero, [0.2, 99.8])

                l_ch_nonzero = np.clip(l_ch_nonzero, l_min_max[0], l_min_max[1])
                a_ch_nonzero = np.clip(a_ch_nonzero, a_min_max[0], a_min_max[1])
                b_ch_nonzero = np.clip(b_ch_nonzero, b_min_max[0], b_min_max[1])

                l_buoy_area = max(l_ch_nonzero.size, 1)
                a_buoy_area = max(a_ch_nonzero.size, 1)
                b_buoy_area = max(b_ch_nonzero.size, 1)
                avg_l = np.sum(l_ch_nonzero) / l_buoy_area
                avg_a = np.sum(a_ch_nonzero) / a_buoy_area
                avg_b = np.sum(b_ch_nonzero) / b_buoy_area

                if self.options['debug']:
                    int_pt = lambda pt: (int(pt[0]), int(pt[1]))
                    tag(avgMat, str("({},{},{})".format(int(avg_l), int(avg_a), int(avg_b))), int_pt(buoy_candidate.center))

                #print(avg_l, avg_a, avg_b)
                buoy_results.append((buoy_candidate, (avg_l, avg_a, avg_b), buoy_candidate.area))

            if self.options['debug']:
                self.post('avg LAB', avgMat)

            # Best fit classification
            [buoy.reset_frame_state() for buoy in BUOYS]
            total_error, buoys_classified, classification = self.new_classify_buoys(BUOYS, buoy_results)
            #print(buoys_classified)
            for (buoy, contour) in classification.items():
                if buoy is None:
                    continue
                if contour is not None:
                    buoy.contour = contour
                    buoy.set_results(mat.shape[1], mat.shape[0], contourScores)
                else:
                    buoy.reset_frame_state()
                    buoy.zero_results()

            buoy_contoursMat = mat.copy()
            for buoy in BUOYS:
                if buoy.contour is not None:
                    cv2.drawContours(buoy_contoursMat, [buoy.contour.contour], -1, buoy.color, 6)

            self.post("total_mask", total_mask)
            self.post("All buoys", buoy_contoursMat)

        else:
            [buoy.zero_results() for buoy in BUOYS]

        for buoy in BUOYS:
            self.fill_single_camera_direction(buoy.results)
            buoy.set_shm_group()
        end_time = time.time()
        #print("Elapsed time: " + str(end_time - start_time))

    def new_classify_buoys(self, buoys, candidates):
        """
            buoys is a list of BuoyData objects
            candidates is a list of (candidate, color) tuples
        """
        # Possible mappings for the # of candidates to the 3 buoys
        buoys = list(buoys)
        while len(candidates) > len(buoys):
            buoys.append(None)
        possible_mappings = permutations(buoys, len(candidates))
        assignments = []
        max_buoys_classified = 0

        for mapping in possible_mappings:
            total_error = 0
            classification = {buoy: None for buoy in mapping if buoy is not None}
            buoys_classified = 0
            # Iterate through one-to-one mappings of buoys to candidates
            for i in range(len(candidates)):
                error = self.new_calculate_error(mapping[i], candidates[i])
                if error >= INVALID_BUOY_ERROR:
                    classification[mapping[i]] = None
                else:
                    classification[mapping[i]] = candidates[i][0] #classification(BUOY COLOR) = tested buoy
                    total_error += error
                    buoys_classified += 1

            if buoys_classified > 0 and total_error < self.options['max_error'] and buoys_classified >= max_buoys_classified and self.check_pos(mapping, classification):

                max_buoys_classified = buoys_classified
                assignments.append((total_error, buoys_classified, classification))

        # Consider assignments with the most number of classified buoys
        assignments = filter(lambda x: x[1] == max_buoys_classified, assignments)
        min_error = self.options['max_error']
        optimal_assignment = (0, 0, {buoy: None for buoy in buoys})
        for assignment in assignments:
            #print(assignment[0])
            if assignment[0] < min_error:
                min_error = assignment[0]
                optimal_assignment = assignment
        #print(len(optimal_assignment[2]), min_error)
        for (k, v) in optimal_assignment[2].items():
            if v is not None:
                for candidate in enumerate(candidates):
                    if candidate[1][0].center == v.center:
                        #print(k.color, candidate[1][1])
                        break
        return optimal_assignment

    def new_calculate_error(self, buoy, candidate):
        if buoy is None:
            return INVALID_BUOY_ERROR
        if candidate[2] == 0:
            return INVALID_BUOY_ERROR
        error_vec = np.array(buoy.reference_color) - \
                    np.array(candidate[1])
        buoy_error = error_vec.dot(error_vec)
        size_error = 1000 / candidate[2]
        if buoy_error > self.options['max_buoy_error']:
            return INVALID_BUOY_ERROR
        #if abs(buoy.results.center_y - 600) > self.options['max_dist_from_center']:
        #    return INVALID_BUOY_ERROR
        else:
            return buoy_error + size_error

if __name__ == '__main__':
    Buoys('forward', options)()
