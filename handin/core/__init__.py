"""Shared, strategy-free building blocks for any agent. See `handin/core/README.md`.

- `camera`:    camera model (both cameras), depth -> 3D points, detection -> world point
- `estimator`: attitude, heading, position, velocity from gyro + accel + VIO (no attitude is given)
- `sticks`:    desired world velocity + heading -> (roll, pitch, yaw, throttle) sticks

Everything here encodes facts about the kit (sensor specs, stick mapping, drag), not choices
about how to search or fly. Optional: an attempt may use it, extend it, or ignore it.
"""
from .camera import camera_pose, depth_points, localise, world_to_pixel, pixel_ray_cam, cam_to_pixel
from .estimator import Estimator, HeadingInit
from .sticks import VelocityController
