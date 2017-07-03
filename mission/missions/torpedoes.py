from collections import namedtuple
import shm
from mission.constants.config import torpedoes as constants
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
    RelativeToCurrentDepth,
    VelocityX,
    VelocityY,
    RelativeToCurrentHeading,
    Depth,
)
from mission.framework.timing import Timer
from mission.framework.primitive import (
    Zero,
    NoOp,
    Log,
    Succeed,
    Fail,
    FunctionTask,
)
from mission.framework.actuators import FireActuator
from mission.framework.position import GoToPosition, MoveX, MoveY
from mission.framework.track import (
    Matcher,
    Match,
    CameraCoord,
    ConsistentObject,
)
from mission.missions.ozer_common import (
    ConsistentTask,
    CenterCentroid,
    AlignHeadingToAngle,
    Except,
    GlobalTimeoutError,
    StillHeadingSearch,
    FailOnExcept,
)

"""
Torpedoes 2017!
"""

class Vision(Task):
    # Board/cutout IDs are indices
    LEFT_ID = 0
    RIGHT_ID = 1
    BOARD_NAMES = ['left', 'right']

    class Obs(CameraCoord):
        def __init__(self, shm_obj, x, y):
            super().__init__(x, y)
            self.adopt_attrs(shm_obj)

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Vision object needs to be ready for rest of mission to access, so must
        # initialize before on_first_run()

        shm.vision_debug.color_r.set(0)
        shm.vision_debug.color_g.set(200)
        shm.vision_debug.color_b.set(255)

        self.boards_matcher = Matcher([
            Match(self.LEFT_ID, CameraCoord(-0.5, 0)),
            Match(self.RIGHT_ID, CameraCoord(0.5, 0)),
        ])

        self.cutout_cons_obj = ConsistentObject()

        self.watcher = watcher()
        self.watcher.watch(shm.bins_vision)
        self.pull()

    def on_run(self, *args, **kwargs):
        if self.watcher.has_changed():
            self.pull()

    def pull(self):
        observations = []
        for i in range(2):
            shm_obj = shm._eval('torpedoes_board{}'.format(i))
            if shm_obj.visible:
                observations.append(self.Obs(shm_obj, shm_obj.x, shm_obj.y))

        self.boards = self.boards_matcher.match(observations)

        cutout_obj = shm.torpedoes_cutout.get()
        noodle_obj = shm.torpedoes_noodle.get()
        cutout_obs = cutout_obj if cutout_obj.visible else None
        noodle_obs = noodle_obj if noodle_obj.visible else None
        self.cutout = Match(0, cutout_obs)
        self.noodle = Match(0, noodle_obs)

        # Debug locations
        for i, obj in enumerate(self.boards + [self.cutout, self.noodle]):
            debug_info_g = shm._eval('vision_debug{}'.format(i))
            debug_info = debug_info_g.get()
            if obj.obs is None:
                debug_info.text = bytes('', 'utf8')
            else:
                obj_type = 'board' if obj in self.boards else 'cutout'
                if obj.id is not None:
                    id_name = 'Left' if obj.id == self.LEFT_ID else 'Right'
                    debug_info.text = bytes('{} {}'.format(id_name, obj_type), 'utf8')
                else:
                    debug_info.text = bytes('{} ({})'.format(i, obj_type), 'utf8')
                debug_info.x, debug_info.y = bin.obs.x, bin.obs.y

            debug_info_g.set(debug_info)

# Forecam offsets may be incorrect to compensate for inaccurate DVL-less control
class Torp:
    FIRE_TIME = 0.5

    def __init__(self, forecam_offset, actuator):
        self.forecam_offset = forecam_offset
        self.actuator = actuator

    def AlignFromForecam(self):
        return Sequential(
            RelativeToCurrentDepth(-self.forecam_offset[1]),
            Timer(0.5),
            MoveY(-self.forecam_offset[0]),
            Timer(0.5),
        )

    def Fire(self):
        return FireActuator(self.actuator, self.FIRE_TIME),

TORPS = [Torp((-0.1, -0.5), 'left_torpedo'), Torp((-0.1, -0.5), 'right_torpedo')]

Cutout = namedtuple('Cutout', ['name', 'obj_func', 'is_noodle'])
CUTOUT = Cutout('top left', lambda vision: vision.cutout, False)
NOODLE = Cutout('bottom right noodle', lambda vision: vision.noodle, True)

class Torpedoes(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Except(
            Sequential(
                Log('Starting torpedoes!'),
                Succeed(TryCompleteCutout(vision, CUTOUT, TORPS[0])),
                Succeed(TryCompleteCutout(vision, NOODLE, TORPS[1])),
            ),

            Fail(Log('Global timeout, aborting torpedoes')),

            GlobalTimeoutError,
        )),

class TryCompleteCutout(Task):
    def on_first_run(self, vision, cutout, torp, *args, **kwargs):
        self.use_task(Conditional(
            Sequential(
                Log('Starting to attempt {} cutout'.format(cutout.name)),
                Retry(CompleteCutout(vision, cutout, torp), 3),
            ),

            on_fail=Fail(Sequential(
                Log('Failed to ever complete {} cutout, firing torpedo anyway'.format(cutout.name)),
                torp.Fire(),
            )),
        ))

class CompleteCutout(Task):
    def on_first_run(self, vision, cutout, torp, *args, **kwargs):
        self.use_task(Sequential(
            Log('Attempting {} cutout'.format(cutout.name)),

            Conditional(
                Retry(MoveInFrontOfBoards(vision), 3),
                on_fail=Fail(Log('Failed to ever move in front of boards')),
            ),

            RestorePos(Sequential(
                AlignCutout(cutout, torp),
                torp.Fire(),
            )),
        ))

class MoveInFrontOfBoards(Task):
    def on_first_run(self, vision, *args, **kwargs):
        self.use_task(Sequential(
            Log('Searching for boards'),
            MasterConcurrent(IdentifyBoard(vision), StillHeadingSearch()),

            AlignBoards(vision),
        ))

class IdentifyBoard(Task):
    def on_run(self, vision, *args, **kwargs):
        if sum(b.obs is not None for b in vision.boards) > 0:
            self.finish()

class AlignBoards(Task):
    """
    Imprecisely align to both torpedoes boards.

    Pre: at least one board in sight
    Post: both boards centered in front of sub
    """

    def on_first_run(self, vision, *args, **kwargs):
        def avg_board_obs(vision):
            for i, b in enumerate(vision.boards):
                if b.obs is not None:
                    avg_i = i
                    avg_b = b
                    break

            other_b = vision.boards[1].obs
            if avg_i == 0 and other_b is not None:
                for attr in ['x', 'y', 'skew', 'width', 'height']:
                    setattr(
                        avg_b,
                        attr,
                        (getattr(avg_b, attr) + getattr(other_b, attr)) / 2,
                    )

                avg_b.clipping = avg_b.clipping or other_b.clipping

            return avg_b

        self.task = Sequential(
            Concurrent(
                AlignSurge(avg_board_obs),

                # Align depth
                PIDLoop(
                    input_value=lambda: avg_board_obs().y,
                    output_function=RelativeToCurrentDepth(),
                    target=0,
                    deadband=0.05,
                    p=0.5,
                ),

                # Align sway (with skew)
                PIDLoop(
                    input_value=lambda: avg_board_obs().skew,
                    output_function=VelocityY(),
                    target=0,
                    deadband=0.05,
                    p=0.5,
                ),

                # Align heading
                PIDLoop(
                    input_value=lambda: avg_board_obs().x,
                    output_function=RelativeToCurrentHeading(),
                    target=0,
                    deadband=0.05,
                    p=0.5,
                ),

                finite=False,
            ),

            Zero(),
        )

    def on_run(self, vision, *args, **kwargs):
        if sum(b.obs is not None for b in vision.boards) == 0:
            self.loge('No boards visible, aborting align')
            self.finish(success=False)
        else:
            self.task()
            if self.task.finished:
                self.finish(success=self.task.success)

class AlignSurge(Task):
    DESIRED_WIDTH = 0.3
    DEADBAND = 0.03
    P = 0.5

    def on_first_run(self, board_obs_func, *args, **kwargs):
        def current_width():
            obs = board_obs_func()
            return self.DESIRED_WIDTH - 0.1 if obs.clipping else obs.width

        self.use_task(PIDLoop(
            input_value=current_width,
            output_function=VelocityX(),
            target=self.DESIRED_WIDTH,
            deadband=self.DEADBAND,
            p=self.P,
        ))

class RestorePos(Task):
    """
    Restore the position of the sub from before the given task started.
    """
    def on_first_run(self, task, *args, **kwargs):
        k = shm.kalman.get()

        self.use_task(Defer(task, Sequential(
            Log('Restoring position'),
            GoToPosition(
                north=k.north,
                east=k.east,
                heading=k.heading,
                depth=k.depth,
            ),
        )))

class AlignCutout(Task):
    SURGE_DISTANCE = 0.5
    NOODLE_SWAY_DISTANCE = 0.75

    def on_first_run(self, cutout, torp, *args, **kwargs):
        def cutout_coord(cutout):
            obj = cutout.obj_func()
            return (obj.obs.x, obj.obs.y)

        def unsee_cutout():
            self.must_see_cutout = False

        self.must_see_cutout = True

        if cutout.is_noodle:
            self.task = Sequential(
                Log('Going down to help find noodle'),
                RelativeToCurrentDepth(0.7),

                Log('Aligning noodle and torpedo tube with cutout depth'),
                Concurrent(
                    # Align horizontally with noodle
                    CenterCentroid(lambda: (cutout_coord()[0], 0), precision=2),
                    Depth(constants.noodle_cutout_depth),
                    finite=False,
                ),
                Zero(),

                FunctionTask(unsee_cutout),

                Log('Pushing away noodle'),
                MoveY(self.NOODLE_SWAY_DISTANCE),
                MoveX(self.SURGE_DISTANCE),
                MoveY(-self.NOODLE_SWAY_DISTANCE - torp.forecam_offset[0]),
            )

        else:
            self.task = Sequential(
                Log('Centering cutout'),
                CenterCentroid(cutout_coord, precision=2),
                Zero(),

                FunctionTask(unsee_cutout),

                Log('Moving close to cutout'),
                MoveX(self.SURGE_DISTANCE),

                Log('Aligning torpedo tube'),
                torp.AlignFromForecam(),
            )

    def on_run(self, cutout, torp, *args, **kwargs):
        if self.must_see_cutout and cutout.obj_func().obs is None:
            self.loge('Cutout lost, cannot align')
            self.finish(success=False)

        else:
            self.task()
            if self.task.finished:
                self.finish(success=self.task.success)
