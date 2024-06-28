import numpy as np


class Ray:
    pos: np.ndarray = np.zeros((3, 1))
    mag: float = 1
    th: float = 0
    ph: float = 0

    def __init__(
        self,
        pos: np.ndarray,  # [x, y, z] and remember, z is the optical axis, the standard viewing axis is the yz axis (x coming out at you)
        mag: float = 0,  # [-] magnification
        th: float = 0,  # [rad] angle about the x-axis
        ph: float = 0,  # [rad] angle about the y-axis
    ):
        self.pos = pos
        self.mag = mag
        self.th = th
        self.ph = ph

    def __repr__(self):
        return f"[{self.pos}; [{self.th}, {self.ph}]]"
