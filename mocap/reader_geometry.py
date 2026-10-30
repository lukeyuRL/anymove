"""One shared letterbox transform for training and detection (pure stdlib).

Image pixels are in the rendered view (which can be scaled relative to layout).
Model pixels are in the square SIZE x SIZE DINOv2 input. The inverse returns
the original 1280x720 front/side *layout* pixel coordinates, never image pixels.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    input_size: int

    def __post_init__(self):
        if any(type(n) is not int or n <= 0
               for n in (self.width, self.height, self.input_size)):
            raise ValueError("letterbox dimensions must be positive integers")

    @property
    def side(self):
        return max(self.width, self.height)

    @property
    def ox(self):
        return (self.side - self.width) // 2

    @property
    def oy(self):
        return (self.side - self.height) // 2

    @property
    def affine(self):
        k = self.input_size / self.side
        return ((k, 0., self.ox * k), (0., k, self.oy * k))

    def to_reader(self, u, v, render_scale=1.):
        self._check_scale(render_scale)
        k = self.input_size / self.side
        return ((u * render_scale + self.ox) * k,
                (v * render_scale + self.oy) * k)

    def to_layout(self, u, v, render_scale=1.):
        self._check_scale(render_scale)
        k = self.side / self.input_size
        return ((u * k - self.ox) / render_scale,
                (v * k - self.oy) / render_scale)

    @staticmethod
    def _check_scale(render_scale):
        if (type(render_scale) not in (float, int) or not
                math.isfinite(render_scale) or render_scale <= 0):
            raise ValueError("render_scale must be finite and positive")
