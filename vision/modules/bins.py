#!/usr/bin/env python3

import time
import cv2
import numpy as np

import shm
from vision import options
from vision.vision_common import (
    draw_angled_arrow,
    get_angle_from_rotated_rect,
    Hierarchy,
    is_clipping,
)
from vision.modules.base import ModuleBase

CONTOUR_HEURISTIC_LIMIT = 5
CONTOUR_SCALED_HEURISTIC_LIMIT = 2

options = [
    options.BoolOption('debug', False),
    options.IntOption('max_fps', 30, 0, 30),
    options.IntOption('c', -60, -255, 255),
    options.IntOption('block_size', 401, 0, 4000, lambda x: x % 2 == 1),
    options.IntOption('thresh_value', 190, 0, 255),
    options.IntOption('morph_size', 10, 1, 30),
    options.DoubleOption('min_size', 0.1, 0, 2), # Min length of min length side
    options.DoubleOption('min_rectangularity', 0.7, 0, 1),
    options.DoubleOption('min_inner_outer_ratio', 0.3, 0, 1),
    options.DoubleOption('cover_threshold', 150, 0, 255),
]

class Bins(ModuleBase):
    def post(self, *args, **kwargs):
        if self.options['debug']:
            super().post(*args, **kwargs)

    def draw_contours(self, mat, *contours):
        cv2.drawContours(mat, contours, -1, (0, 127, 255), thickness=3)

    def process(self, mat):
        start_time = time.time()

        self.process_bins(mat)
        shm.bins_vision.clock.set(not shm.bins_vision.clock.get())

        runtime = time.time() - start_time
        min_runtime = 1 / self.options['max_fps']
        if min_runtime > runtime:
            time.sleep(min_runtime - runtime)
            runtime = min_runtime
        print('FPS: {}'.format(1 / (runtime)))

    def process_bins(self, mat):
        results = [shm.bins_bin0.get(), shm.bins_bin1.get()]
        for result in results:
            result.visible = False

        self.post('orig', mat)

        self.bgr_b, self.bgr_g, self.bgr_r = cv2.split(mat)
        _, threshed = cv2.threshold(
            self.bgr_b,
            self.options['thresh_value'],
            255,
            cv2.THRESH_BINARY,
        )
        self.post('threshed', threshed)

        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.options['morph_size'],) * 2
        )
        # Get rid of small things
        morphed = cv2.erode(threshed, morph_kernel)
        # Fill back in holes we just created
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, morph_kernel)
        self.post('morphed', morphed)

        _, contours, hierarchy = cv2.findContours(
            morphed.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None:
            hierarchy = [[]]
        hier = Hierarchy(hierarchy)

        outer_contours = [{'i': i, 'contour': contours[i]} for i in hier.outermost()]

        big_rects = []
        for info in outer_contours:
            info['rect'] = cv2.minAreaRect(info['contour'])
            if min(info['rect'][1]) / len(mat[1]) < self.options['min_size']:
                continue

            rectangularity = cv2.contourArea(info['contour']) / np.prod(info['rect'][1])
            if rectangularity < self.options['min_rectangularity']:
                continue

            if is_clipping(mat, info['contour']):
                continue

            big_rects.append(info)

        concentric_rects = []
        for info in big_rects:
            child_i = hier.first_child(info['i'])
            if child_i == -1:
                continue

            max_child_i = max(
                hier.siblings(child_i),
                key=lambda x: cv2.contourArea(contours[x])
            )
            info['inner_contour'] = contours[max_child_i]
            info['inner_rect'] = cv2.minAreaRect(info['inner_contour'])
            inner_area = cv2.contourArea(info['inner_contour'])
            rectangularity = inner_area / (info['inner_rect'][1][0] * info['inner_rect'][1][1])
            if rectangularity < self.options['min_rectangularity']:
                continue

            inner_size_ratio = min(info['inner_rect'][1]) / min(info['rect'][1])
            if inner_size_ratio < self.options['min_inner_outer_ratio']:
                continue

            inner_mask = np.zeros(mat.shape[:2], dtype=np.uint8)
            cv2.drawContours(inner_mask, [info['inner_contour']], -1, 255, -1)
            average_r = cv2.mean(self.bgr_r, inner_mask)[0]
            info['covered'] = average_r > self.options['cover_threshold']

            info['angle'] = get_angle_from_rotated_rect(info['rect'])

            concentric_rects.append(info)

        if self.options['debug']:
            contours_mat = mat.copy()
            self.draw_contours(contours_mat, *[info['contour'] for info in concentric_rects])
            self.draw_contours(contours_mat, *[info['inner_contour'] for info in concentric_rects])
            for info in concentric_rects:
                draw_angled_arrow(contours_mat, info['rect'][0], info['angle'])
                if info['covered']:
                    cv2.drawContours(
                        contours_mat,
                        [info['inner_contour']],
                        -1,
                        (20, 255, 57),
                        thickness=10,
                    )
            self.post('contours', contours_mat)

        concentric_rects.sort(key=lambda x: -x['rect'][1][0] * x['rect'][1][1])

        for info, result in zip(concentric_rects, results):
            result.visible = True
            result.clipping = False
            result.x, result.y = self.normalized(info['inner_rect'][0])
            result.width, result.length = self.normalized_size(sorted(info['rect'][1]))
            result.angle = info['angle']
            result.covered = info['covered']

        shm.bins_bin0.set(results[0])
        shm.bins_bin1.set(results[1])

if __name__ == '__main__':
    Bins('downward', options)()
