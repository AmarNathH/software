import math
import itertools
import shm
from mission.framework.task import Task
from mission.framework.targeting import DownwardTarget
from mission.framework.combinators import Sequential
from mission.framework.timing import Timer, Timeout
from mission.framework.movement import (
    RelativeToCurrentHeading,
    RelativeToCurrentDepth,
    Heading,
)
from mission.framework.helpers import call_if_function, ConsistencyCheck
from mission.framework.search import SpiralSearch
from mission.framework.primitive import InvertSuccess
from auv_python_helpers.angles import heading_sub_degrees

"""
Tasks that I (Ozer) like to use in missions but don't feel are worthy of
necessary adding directly to the mission framework just yet.
"""

class ConsistentTask(Task):
    """
    Finishes when a non-finite task is consistently finished
    """
    def on_first_run(self, task, success=18, total=20, *args, **kwargs):
        self.cons_check = ConsistencyCheck(success, total)
        
    def on_run(self, task, *args, **kwargs):
        task()
        if self.cons_check.check(task.finished):
            self.finish()

class CenterCentroid(Task):
    """
    Center the centroid of all provided objects in the downcam

    Begin: at least one object in view
    End: center of all objects in center of camera

    args:
        points_func: function which returns a list of (x, y) tuples
        corresponding to the coordinates of the points. Its output should not
        vary within a single tick.
    """
    PS = [1.0, 1.0, 0.6]
    DS = [0.3, 0.3, 0.2]
    # DEADBANDS = [(0.1, 0.1), (0.0375, 0.0375), (0.0375, 0.0375)]
    DEADBANDS = [(0.15, 0.15), (0.0675, 0.0675), (0.0475, 0.0475)]
    MAX_SPEED = 0.25

    def centroid(self, points_func):
        points = points_func()
        center_x, center_y = 0, 0
        for x, y in points:
            center_x += x
            center_y += y

        return (center_x / len(points), center_y / len(points))

    def on_first_run(self, points_func, target=(0, 0), precision=0, *args, **kwargs):
        self.task = ConsistentTask(DownwardTarget(
            point=lambda: self.centroid(points_func),
            target=target,
            deadband=self.DEADBANDS[precision],
            px=self.PS[precision],
            py=self.PS[precision],
            dx=self.DS[precision],
            dy=self.DS[precision],
            max_out=self.MAX_SPEED,
        ))

    def on_run(self, points_func, *args, **kwargs):
        if len(points_func()) == 0:
            self.loge("Can't see any objects, targeting aborted")
            self.finish(success=False)

        else:
            self.task()
            if self.task.finished:
                self.finish()

class GradualHeading(Task):

    class GradualApproxHeading(Task):
        RELATIVE_DESIRE = 15
        DEADBAND = 25

        def on_first_run(self, *args, **kwargs):
            self.rel_curr_heading = RelativeToCurrentHeading()

        def on_run(self, desire, *args, **kwargs):
            current = shm.kalman.heading.get()
            diff = heading_sub_degrees(call_if_function(desire), current)
            if abs(diff) < self.DEADBAND:
                self.finish()
                return

            relative = math.copysign(self.RELATIVE_DESIRE, diff)
            self.rel_curr_heading(relative)

    def on_first_run(self, desire, *args, **kwargs):
        self.use_task(Sequential(
            self.GradualApproxHeading(desire),
            Heading(desire),
            finite=False,
        ))

class GradualDepth(Task):
    RELATIVE_DESIRE = 0.18
    DEADBAND = 0.08

    def on_run(self, depth, *args, **kwargs):
        diff = call_if_function(depth) - shm.kalman.depth.get()
        relative = math.copysign(self.RELATIVE_DESIRE, diff)
        RelativeToCurrentDepth(relative)()
        if abs(diff) < self.DEADBAND:
            self.finish()

class AlignHeadingToAngle(Task):
    """
    Align the sub's heading to make the given angle appear at a target angle.

    The angle is from objects beneath the sub, is in degrees, starts at 0 from the
    right, and wrap positively counterclockwise (like a standard math angle).
    """

    def on_first_run(self, current, desire, mod=360, *args, **kwargs):
        def desired_heading():
            return shm.kalman.heading.get() - (heading_sub_degrees(
                call_if_function(desire),
                call_if_function(current),
                mod=mod
            ))

        self.use_task(Sequential(
            # GradualHeading(desired_heading),
            Heading(desired_heading),
            finite=False,
        ))

class GlobalTimeoutError(Exception):
    pass

# TODO improve speed
# Pitch/roll searching? Navigation spline?
# TODO make work without positional control too
class SearchWithGlobalTimeout(Task):
    def on_first_run(self, *args, **kwargs):
        self.use_task(Timeout(Sequential(
            # Pause initially to give object-identifying tasks time to check current state
            Timer(0.5),

            SpiralSearch(
                relative_depth_range=0,
                optimize_heading=True,
                meters_per_revolution=1,
                min_spin_radius=1,
            )
        ), 120))

    def on_finish(self, *args, **kwargs):
        if not self.success:
            self.loge('Timed out while searching')
            raise GlobalTimeoutError()

class Except(Task):
    def on_first_run(self, *args, **kwargs):
        self.excepted = False

    def on_run(self, main_task, except_task, *exceptions, **kwargs):
        if not self.excepted:
            try:
                main_task()
                if main_task.finished:
                    self.finish(success=main_task.success)

                return

            except exceptions:
                self.excepted = True

        except_task()
        if except_task.finished:
            self.finish(success=except_task.success)

class ConsistentTask(Task):
    """
    Finishes when a non-finite task is consistently finished
    """
    def on_first_run(self, task, success=18, total=20, *args, **kwargs):
        self.cons_check = ConsistencyCheck(success, total)

    def on_run(self, task, *args, **kwargs):
        task()
        if self.cons_check.check(task.finished):
            self.finish()

class Disjunction(Task):
    """
    Run tasks in order as they fail, and succeed when the first task does. Fail
    if no task succeeds.

    Disjunction is to Sequential as 'or' is to 'and'.
    """
    def on_first_run(self, *tasks, subtasks=(), finite=True, **kwargs):
        self.use_task(InvertSuccess(Sequential(
            subtasks=[InvertSuccess(t) for t in itertools.chain(tasks, subtasks)]
        )))


class StillHeadingSearch(Task):
    """
    Search for an object visible from the current location that is in front of
    the sub with highest probability.
    """

    TIMEOUT = 120

    def on_first_run(self, *args, **kwargs):
        init_heading = shm.kalman.heading.get()

        self.use_task(Timed(
            While(lambda: Sequential(
                # Pause a little to let object-recognizing tasks see the current fov
                Timer(0.5),

                # Check the right side
                GradualHeading(init_heading + 90),
                Timer(0.5),
                Heading(init_heading),

                # Check the left and back
                GradualHeading(init_heading - 90),
                GradualHeading(init_heading - 180),
                GradualHeading(init_heading - 270),
                Timer(0.5),
                Heading(init_heading),

                # Move back a bit, we might be too close
                MoveX(-1),
            ), True),

            self.TIMEOUT,
        ))

    def on_finish(self, *args, **kwargs):
        self.loge('Timed out while searching')
        raise GlobalTimeoutError()
