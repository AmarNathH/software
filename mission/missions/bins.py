import shm
from shm.watchers import watcher
from mission.framework.task import Task
from mission.framework.targeting import PIDLoop
from mission.framework.combinators import (
    Sequential,
    Concurrent,
    MasterConcurrent,
    Retry,
    Conditional,
    Defer,
)
from mission.framework.movement import (
    Depth,
    RelativeToInitialDepth,
    RelativeToCurrentDepth,
    VelocityY,
)
from mission.framework.timing import Timer, Timeout, Timed
from mission.framework.primitive import (
    Zero,
    FunctionTask,
    NoOp,
    Log,
    Succeed,
    Fail,
)
from mission.framework.actuators import FireActuator
from mission.framework.position import MoveXY, PositionalControl
from mission.framework.track import (
    Matcher,
    Match,
    Observation,
    HeadingInvCameraCoord,
)
from mission.missions.ozer_common import (
    ConsistentTask,
    CenterCentroid,
    AlignHeadingToAngle,
    SearchWithGlobalTimeout,
    Except,
    GlobalTimeoutError,
)

"""
Bins 2017!
"""

class Vision(Task):
    # Bin IDs
    TARGET_BIN = 0 # The bin that was originally covered (even if not currently)
    OTHER_BIN = 1 # The bin that was never covered

    class BinObs(HeadingInvCameraCoord):
        def __init__(self, shm_bin, heading):
            super().__init__(shm_bin.x, shm_bin.y, heading)
            self.adopt_attrs(shm_bin)

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Vision object needs to be ready for rest of mission to access, so must
        # initialize before on_first_run()

        shm.vision_debug.color_r.set(0)
        shm.vision_debug.color_g.set(200)
        shm.vision_debug.color_b.set(255)

        self.bins_matcher = Matcher(
            [], # We don't know the bins' relative positions ahead of time
            num_trackers=2,
        )

        self.watcher = watcher()
        self.watcher.watch(shm.bins_vision)
        self.pull()

    def on_run(self, *args, **kwargs):
        if self.watcher.has_changed():
            self.pull()

    def classify(self):
        if sum(bin.obs is not None for bin in self.bins) < 2:
            self.loge('Failed to classify, not all bins visible')
            return False

        covered_indices = [i for i, bin in enumerate(self.bins) if bin.obs.covered]
        if len(covered_indices) != 1:
            self.loge('Failed to classify, single covered bin not found')
            return False

        covered_i = covered_indices[0]

        self.bins = self.bins_matcher.update_pattern([
            Match(self.TARGET_BIN, self.bins[covered_i].obs),
            Match(self.OTHER_BIN, self.bins[1-covered_i].obs),
        ])
        return True

    def pull(self):
        shm_bins = [
            shm.bins_bin0.get(),
            shm.bins_bin1.get(),
        ]
        heading = shm.kalman.heading.get()
        observations = [self.BinObs(sbin, heading) for sbin in shm_bins if sbin.visible]
        self.bins = self.bins_matcher.match(observations)

        # Debug locations
        for i, bin in enumerate(self.bins):
            debug_info_g = shm._eval('vision_debug{}'.format(i))
            debug_info = debug_info_g.get()
            if bin.obs is None:
                debug_info.text = bytes('', 'utf8')
            else:
                if bin.id is not None:
                    debug_info.text = bytes('Target bin' if bin.id == self.TARGET_BIN else 'Other bin', 'utf8')
                else:
                    debug_info.text = bytes('Bin {}'.format(i), 'utf8')
                debug_info.x, debug_info.y = bin.obs.x, bin.obs.y

            debug_info_g.set(debug_info)

# TODO get actual grabber actuators
def CloseGrabber():
    return Sequential(
        Log('Closing grabber'),
        # FireActuator("piston_extend", 0.4)
    )

def OpenGrabber():
    return Sequential(
        Log('Opening grabber'),
        # FireActuator("piston_retract", 0.4),
    )

class Bins(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Conditional(
            Except(Sequential(
                Log('Starting bins'),

                OpenGrabber(),

                Conditional(
                    Retry(lambda: ClassifyBins(vision), 3),
                    on_fail=Fail(Log('Failed to ever classify bins')),
                ),

                Conditional(
                    Retry(lambda: Uncover(vision), 3),
                    on_fail=Fail(Log('Failed to ever remove cover')),
                ),

                Conditional(
                    Retry(lambda: Drop(vision), 3),
                    on_fail=Fail(Log('Failed to ever accurately drop markers')),
                ),
            ), Fail(), GlobalTimeoutError),

            Log('Bins success! :O'),

            Fail(Sequential(Zero(), FastDrop(), Log('Bins failure! :('))),
        ))

class ClassifyBins(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Conditional(
            Sequential(
                MoveAboveBins(vision),
                Timer(0.5), # Wait for vision to stabilize
                FunctionTask(vision.classify),
            ),

            Log('Bins classified'),

            Fail(Log('Failed to classify bins')),
        ))

class Uncover(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Conditional(
            Sequential(
                Retry(lambda: MoveAboveBins(vision), 3),
                RemoveCover(vision),
            ),

            Log('Bin uncovered!'),

            Fail(Log('Failed to uncover bin')),
        ))

class MoveAboveBins(Task):
    DEPTH = 0.5

    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Conditional(
            Sequential(
                Log('Moving to depth where bins are visible'),
                Depth(self.DEPTH),

                Log('Searching for bin'),
                MasterConcurrent(IdentifyBin(vision), SearchWithGlobalTimeout()),

                Log('Centering bins'),
                CenterBins(vision),
            ),

            on_fail=Fail(Log('Failed to move above bins')),
        ))

class IdentifyBin(Task):
    def on_run(self, vision, *args, **kwargs):
        if any(bin.obs is not None for bin in vision.bins):
            self.finish()

class CenterBins(Task):
    def on_first_run(self, vision, filter=lambda bin: True, *args, **kwargs):

        def bin_points():
           return [
               (bin.obs.x, bin.obs.y) for bin in vision.bins
               if bin.obs is not None and filter(bin)
           ]

        self.use_task(Conditional(
            CenterCentroid(bin_points),
            on_fail=Fail(Log('Failed to center bins')),
            finite=False,
        )),


class RemoveCover(Task):
    def on_first_run(self, vision, *args, **kwargs):
        def CheckRemoved():
            return FunctionTask(lambda: all(not bin.obs.covered for bin in vision.bins))

        self.use_task(Conditional(
            Sequential(
                MoveAboveBins(vision),
                Conditional(
                    CheckRemoved(),

                    Log('The cover is already gone?!?'),

                    Sequential(
                        AlignOverTargetBin(vision),
                        PullOffCover(),
                        Log('Verifying the cover was removed'),
                        MoveAboveBins(vision),
                        CheckRemoved(),
                    ),
                ),
            ),

            on_fail=Fail(Log('Failed to remove cover')),
        ))

class AlignOverTargetBin(Task):
    def get_target_bin(self, vision):
        targets = [bin for bin in vision.bins if bin.id == vision.TARGET_BIN]
        if len(targets) == 1:
            return targets[0]
        else:
            return None

    def on_first_run(self, vision, *args, **kwargs):

        def CenterTargetBin():
            return CenterBins(vision, lambda bin: bin.id == vision.TARGET_BIN)

        def AlignTargetBin():
            return AlignHeadingToAngle(lambda: self.get_target_bin(vision).obs.angle, 90, mod=180)

        self.task = Sequential(
            Log('Centering over target bin'),
            CenterTargetBin(),

            Log('Aligning target bin'),
            AlignTargetBin(),

            Log('Going down to precisely align with target bin'),
            Concurrent(
                CenterTargetBin(),
                AlignTargetBin(),
                AlignBinDepth(vision, lambda: self.get_target_bin(vision)),
                finite=False,
            ),
            Zero(),
            PositionalControl(),
        )

    def on_run(self, vision, *args, **kwargs):
        if self.get_target_bin(vision) == None:
            self.loge('Failed to align over target bin, lost bin')
            self.finish(success=False)
            Zero()()

        else:
            self.task()
            if self.task.finished:
                self.finish(success=self.task.success)

class AlignBinDepth(Task):
    def on_first_run(self, vision, bin_func, *args, **kwargs):
        self.use_task(ConsistentTask(PIDLoop(
            input_value=lambda: bin_func().obs.length,
            output_function=RelativeToCurrentDepth(),
            target=1.5,
            deadband=0.02,
            p=1.2,
            d=0.5,
        )))

class PullOffCover(Task):
    PICKUP_DELTA_DEPTH = 0.1
    DELTA_DEPTH_TIMEOUT = 4
    SLIDE_TIME = 2
    SLIDE_SPEED = 0.5
    TOTAL_TIMEOUT = 60

    def on_first_run(self, *args, **kwargs):
        self.use_task(Timeout(
            Sequential(
                Log('Aligning grabber above cover handle'),
                MoveXY((0.5, 0)), # TODO get actual grabber offset

                Log('Moving grabber down to touch cover handle'),
                Succeed(Timeout(
                    RelativeToCurrentDepth(self.PICKUP_DELTA_DEPTH),
                    self.DELTA_DEPTH_TIMEOUT,
                )),

                Log('Grabbing bin'),
                CloseGrabber(),
                Timer(1.5),

                Log('Picking up cover slightly'),
                Succeed(Timeout(
                    RelativeToInitialDepth(-self.PICKUP_DELTA_DEPTH),
                    self.DELTA_DEPTH_TIMEOUT,
                )),

                Log('Sliding cover out of the way'),
                Timed(VelocityY(self.SLIDE_SPEED), self.SLIDE_TIME),
                Zero(),

                Log('Dropping cover off here'),
                OpenGrabber(),

                Log('Attempting to return to near pre-grab location'),
                RelativeToInitialDepth(-0.5),
                Timed(VelocityY(-self.SLIDE_SPEED), self.SLIDE_TIME),
                Zero(),
            ), self.TOTAL_TIMEOUT))

class Drop(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Conditional(
            Sequential(
                Log('Starting drop'),
                Retry(lambda: MoveAboveBins(vision), 3),
                AlignOverTargetBin(vision),
                FireMarkers(),
            ),

            on_fail=Fail(Log('Failed to drop')),
        ))

class FireMarkers(Task):
    # TODO get actual marker positions and actuators
    LEFT_MARKER_POS = (0, -0.25)
    RIGHT_MARKER_POS = (0, 0.25)
    FIRE_TIME = 0.1

    def on_first_run(self, in_same_place=True, *args, **kwargs):
        marker1_mv = (-x for x in self.LEFT_MARKER_POS)
        marker2_mv = (x1 - x2 for x1, x2 in zip(self.LEFT_MARKER_POS, self.RIGHT_MARKER_POS))

        self.use_task(Sequential(
            Log('Firing left marker'),
            MoveXY(marker1_mv) if in_same_place else NoOp(),
            # FireActuator('left_marker', self.FIRE_TIME),

            Log('Firing right marker'),
            MoveXY(marker2_mv) if in_same_place else NoOp(),
            # FireActuator('left_marker', self.FIRE_TIME),
        ))

class FastDrop(Task):
    def on_first_run(self, *args, **kwargs):
        self.use_task(Sequential(
            Log('Dropping markers quickly wherever we are now'),
            FireMarkers(in_same_place=False),
        ))

vision = Vision()
def bins(): return MasterConcurrent(Bins(vision), vision)
def test_classify():
    return Concurrent(
        Sequential(Timer(3), FunctionTask(vision.classify)),
        vision,
    )
