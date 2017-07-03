#!/usr/bin/env python3

import math

import shm

from mission.constants.config import PIPE_SEARCH_DEPTH, PIPE_FOLLOW_DEPTH
from mission.framework.combinators import Sequential, Concurrent, Retry
from mission.framework.helpers import get_downward_camera_center, ConsistencyCheck
from mission.framework.movement import Depth, Heading, Pitch
from mission.framework.position import PositionalControl
from mission.framework.primitive import Zero
from mission.framework.search import SearchFor, VelocitySwaySearch, SwaySearch, PitchSearch
from mission.framework.targeting import DownwardTarget, PIDLoop
from mission.framework.task import Task
from mission.framework.timing import Timer

class center(Task):
    def update_data(self):
        self.pipe_results = shm.pipe_results.get()

    def on_first_run(self):
        self.update_data()

        pipe_found = self.pipe_results.heuristic_score > 0

        self.centered_checker = ConsistencyCheck(8, 10)

        self.center = DownwardTarget(lambda self=self: (self.pipe_results.center_x, self.pipe_results.center_y),
                                     target=get_downward_camera_center,
                                     deadband=(30,30), px=0.002, py=0.002, dx=0.002, dy=0.002,
                                     valid=pipe_found)
        self.logi("Beginning to center on the pipe")
        

    def on_run(self):
        self.update_data()
        self.center()
        #self.logi("Results Y: {}".format(str(self.pipe_results.center_y)))
        #self.logi("Center: {}".format(str(get_downward_camera_center()[1])))

        if not check_seen():
            self.finish(success=False)

        if self.centered_checker.check(self.center.finished):
            self.center.stop()
            self.finish()

class align(Task):
    def update_data(self):
        self.pipe_results = shm.pipe_results.get()

    def on_first_run(self):
        self.update_data()

        self.align = Heading(lambda: self.pipe_results.angle + shm.kalman.heading.get(), deadband=0.5)
        self.alignment_checker = ConsistencyCheck(39, 40)

        pipe_found = self.pipe_results.heuristic_score > 0

        self.center = DownwardTarget(lambda self=self: (self.pipe_results.center_x, self.pipe_results.center_y),
                                     target=get_downward_camera_center,
                                     deadband=(10,10), px=0.001, py=0.001, dx=0.002, dy=0.002,
                                     valid=pipe_found)

        self.logi("Beginning to align to the pipe's heading")

    def on_run(self): 
        self.update_data()

        self.align()
        self.center()

        if not check_seen():
            self.finish(success=False)

        if self.alignment_checker.check(self.align.finished):
            self.finish()

search_task = lambda: SearchFor(VelocitySwaySearch(),
                                lambda: shm.pipe_results.heuristic_score.get() > 0,
                                consistent_frames=(6, 6))

pitch_search_task = lambda: SearchFor(PitchSearch(30),
                                      lambda: shm.pipe_results.heuristic_score.get() > 0,
                                      consistent_frames=(6, 6))

pipe_test = lambda: Sequential(Depth(PIPE_SEARCH_DEPTH),
                               search_task(), center(), align(),
                               Depth(PIPE_FOLLOW_DEPTH))

pitch_pipe_test = lambda: Sequential(Depth(PIPE_SEARCH_DEPTH),
                          pitch_search_task(), Zero(),
                          Concurrent(center(), Pitch(0)), Zero(),
                          center(), align(), Depth(PIPE_FOLLOW_DEPTH))

def check_seen():
    visible = shm.pipe_results.heuristic_score.get()
    
    #print(visible)
    if visible > 0:
        return True
    else:
        print('Lost Pipe!')
        return False

def one_pipe(grp):
    return Sequential(
        Depth(PIPE_SEARCH_DEPTH),
        Retry(lambda: Sequential(
             Zero(),
             search_task(),
             center(),
            Concurrent(
             align(),
             center(),
             finite=False
            ),
            ), 100),
        PositionalControl(),
        Zero(),

        Depth(PIPE_FOLLOW_DEPTH)
        )

pipe_mission = one_pipe(shm.desires)

def pitch_pipe(grp):
    return Sequential(
             Depth(PIPE_SEARCH_DEPTH),
             pitch_search_task(),
             Zero(),
             center(),
             Pitch(0),
             center(),

             Concurrent(
                 center(),
                 align(),
                 finite=False,
             ),
             PositionalControl(),
             Zero(),

             Depth(PIPE_FOLLOW_DEPTH))

pitch_pipe_mission = pitch_pipe(shm.desires)

class Timeout(Task):
    def on_first_run(self, time, task, *args, **kwargs):
        self.timer = Timer(time)

    def on_run(self, time, task, *args, **kwargs):
        task()
        self.timer()
        if task.finished:
          self.finish()
        elif self.timer.finished:
          self.logw('Task timed out in {} seconds!'.format(time))
          self.finish()

class OptimizablePipe(Task):
  def desiredModules(self):
    return [shm.vision_modules.Pipes]

  def on_first_run(self, grp):
    self.subtask = one_pipe(grp)
    self.has_made_progress = False

  def on_run(self, grp):
    self.subtask()
    if self.subtask.finished:
      self.finish()
