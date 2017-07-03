from mission.constants.missions import Wire, Recovery, ComplexColor, RecoveryColor, Torpedoes

BUOY_BACKUP_DIST = 2.0
BUOY_BACKUP_FOR_OVER_DIST = 0.2
BUOY_OVER_DEPTH = .8
BUOY_SEARCH_DEPTH = 1.5
BUOY_TO_PIPE_DIST = 2.5

PIPE_SEARCH_DEPTH = 0.4
PIPE_FOLLOW_DEPTH = 0.8

BINS_CAM_TO_ARM_OFFSET = 0.12
BINS_HEIGHT = 0.5
BINS_DROP_ALTITUDE = BINS_HEIGHT + 0.65
BINS_PICKUP_ALTITUDE = BINS_HEIGHT + 0.2
BINS_SEARCH_DEPTH = 0.5

HYDROPHONES_SEARCH_DEPTH = 0.5
HYDROPHONES_PINGER_DEPTH = 3.0

wire = Wire(
    max_distance=4,
    debugging=False,
    thresh_c=8,
    kernel_size=4,
    min_area=200,
    block_size=411,
)

recovery = Recovery(
    tower_depth=0.5,
    tube_dive_altitude=1.7,
    tube_grab_altitude=1.28,
    table_depth=0.5,
    ellipse_depth=2.5,
    drop_depth=3.11,

    detect_table=False, # TODO change to True for Transdec
    table_thresh=100,
    blue_ellipse_thresh=45,
    c0=-15,
    c1=-42,

    colors=[
        # First two are smaller ones
        RecoveryColor(
            name='blue',
            tube_color=ComplexColor(lab_a=178, lab_b=49, ycrcb_cr=115, ycrcb_cb=208),
            ellipse_color=ComplexColor(lab_a=178, lab_b=49, ycrcb_cr=115, ycrcb_cb=208),
        ),
        RecoveryColor(
            name='red',
            tube_color=ComplexColor(lab_a=189, lab_b=165, ycrcb_cr=210, ycrcb_cb=99),
            ellipse_color=ComplexColor(lab_a=189, lab_b=165, ycrcb_cr=210, ycrcb_cb=99),
        ),
        RecoveryColor(
            name='green',
            tube_color=ComplexColor(lab_a=57, lab_b=191, ycrcb_cr=59, ycrcb_cb=73),
            ellipse_color=ComplexColor(lab_a=57, lab_b=191, ycrcb_cr=59, ycrcb_cb=73),
        ),
        RecoveryColor(
            name='orange',
            tube_color=ComplexColor(lab_a=147, lab_b=181, ycrcb_cr=170, ycrcb_cb=72),
            ellipse_color=ComplexColor(lab_a=147, lab_b=181, ycrcb_cr=170, ycrcb_cb=72),
        ),
    ]
)

torpedoes = Torpedoes(
    noodle_cutout_depth=2,
)
