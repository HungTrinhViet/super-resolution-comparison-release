"""Synchronized SR-coordinate viewport; only visible display pixels are built."""
from math import ceil

from PIL import Image


class Viewport:
    def __init__(self):
        self.zoom = 2
        self.offset = (0.0, 0.0)
        self.image_size = (0, 0)
        self.view_size = (1, 1)

    def clamp(self):
        self.offset = tuple(max(0.0, min(value, max(0.0, extent - visible / self.zoom)))
                            for value, extent, visible in zip(self.offset, self.image_size, self.view_size))

    def set_zoom(self, zoom):
        if zoom not in (1, 2, 4, 8):
            raise ValueError('Zoom must be 1, 2, 4 or 8.')
        self.zoom = zoom
        self.clamp()

    def pan(self, dx, dy):
        self.offset = (self.offset[0] - dx / self.zoom, self.offset[1] - dy / self.zoom)
        self.clamp()

    def render(self, image, canvas_size, *, virtual_scale=1):
        """Sample the source directly, including virtual x4 LR, into a bounded tile."""
        width, height = canvas_size
        logical_w, logical_h = image.width * virtual_scale, image.height * virtual_scale
        draw_w = min(width, max(0, ceil((logical_w - self.offset[0]) * self.zoom)))
        draw_h = min(height, max(0, ceil((logical_h - self.offset[1]) * self.zoom)))
        if draw_w < 1 or draw_h < 1:
            return None
        x, y = self.offset
        box = (x / virtual_scale, y / virtual_scale,
               (x + draw_w / self.zoom) / virtual_scale,
               (y + draw_h / self.zoom) / virtual_scale)
        return image.transform((draw_w, draw_h), Image.Transform.EXTENT, box,
                               resample=Image.Resampling.NEAREST)
