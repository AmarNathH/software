from collections import namedtuple

Wire = namedtuple('Wire', [
    'max_distance',
    'debugging',
    'thresh_c',
    'kernel_size',
    'min_area',
    'block_size',
])

Recovery = namedtuple('Recovery', [
    'tower_depth',
    'tube_dive_altitude',
    'tube_grab_altitude',
    'table_depth',
    'ellipse_depth',
    'drop_depth',

    'detect_table',
    'table_thresh',
    'blue_ellipse_thresh',
    'c0',
    'c1',

    'colors',
])

ComplexColor = namedtuple('ComplexColor', [
    'lab_a',
    'lab_b',
    'ycrcb_cr',
    'ycrcb_cb',
])

RecoveryColor = namedtuple('RecoveryColor', [
    'name',
    'tube_color',
    'ellipse_color',
])

Torpedoes = namedtuple('Torpedoes', [
    # Depth of the sub to align torpedo tubes with the noodle-covered cutout
    'noodle_cutout_depth',
])
