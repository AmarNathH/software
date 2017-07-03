#!/usr/bin/env python3
import time

from vision.modules.base import ModuleBase

class Poster(ModuleBase):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  def process(self, *mats):
    for i, im in enumerate(mats):
      self.post(str(i), im)

if __name__ == '__main__':
    Poster()()
