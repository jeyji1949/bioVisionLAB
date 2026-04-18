from dataclasses import dataclass, field
import numpy as np


@dataclass
class AppState:
    original_pixels: np.ndarray = field(default=None)
    img_w: int = 0
    img_h: int = 0
    filepath: str = ""
    filename: str = ""

    m1_result: np.ndarray = field(default=None)
    m2_result: np.ndarray = field(default=None)
    m3_result: np.ndarray = field(default=None)
    m4_result: np.ndarray = field(default=None)
    mc_results: object = field(default=None) 

    m1_log: dict = field(default_factory=dict)
    m2_log: dict = field(default_factory=dict)
    m3_log: dict = field(default_factory=dict)
    m4_log: dict = field(default_factory=dict)

    steps_done: list = field(default_factory=lambda: [False, False, False, False, False])
    on_step_complete: object = field(default=None)

    def has_image(self):
        return self.original_pixels is not None

    def mark_step(self, idx):
        self.steps_done[idx] = True
        if self.on_step_complete:
            self.on_step_complete(idx)

    def pipeline_input(self, step_idx):
        if step_idx == 0:
            return self.original_pixels
        results = [self.m1_result, self.m2_result, self.m3_result]
        for r in reversed(results[:step_idx]):
            if r is not None:
                return r
        return self.original_pixels

    def reset(self):
        self.m1_result = None
        self.m2_result = None
        self.m3_result = None
        self.m4_result = None
        self.mc_results = None
        self.m1_log = {}
        self.m2_log = {}
        self.m3_log = {}
        self.m4_log = {}
        self.steps_done = [False, False, False, False, False]