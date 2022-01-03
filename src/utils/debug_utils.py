import torch
import time
import os

DEBUG = False
if "DEBUG" in os.environ:
    if os.environ["DEBUG"] == "1":
        DEBUG = True


class Timer:
    def __init__(self, name="script"):
        self.name = name

    def __enter__(self):
        if DEBUG:
            torch.cuda.synchronize()
            self.start = time.time()

    def __exit__(self, *args, **kwargs):
        if DEBUG:
            torch.cuda.synchronize()
            print(self.name, time.time() - self.start)
