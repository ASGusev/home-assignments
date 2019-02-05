#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


NEIGHBOUR_DIST = 16
COVER_KERNEL = np.ones((2 * NEIGHBOUR_DIST + 1, 2 * NEIGHBOUR_DIST + 1), dtype=np.uint8)
CORNERS_PER_TIME = 1000
CORNER_QUALITY = 0.005


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _points_coordinates(points, h, w):
    points = points[:, 0, ::-1].T.astype(np.int32)
    points[0][points[0] >= h] = h - 1
    points[1][points[1] >= w] = w - 1
    return tuple(points)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    highlight_size = NEIGHBOUR_DIST
    points = cv2.goodFeaturesToTrack(frame_sequence[0], CORNERS_PER_TIME, CORNER_QUALITY, NEIGHBOUR_DIST)
    track_ids = np.arange(len(points))
    tracks = [[point] for point in points]
    corners_coordinates = np.squeeze(np.int32(points))
    corner_ids = np.arange(len(corners_coordinates))
    corner_sizes = np.zeros(len(corners_coordinates), dtype=np.int32) + highlight_size
    corners = FrameCorners(corner_ids, corners_coordinates, corner_sizes)
    builder.set_corners_at_frame(0, corners)
    prev_image = np.uint8(frame_sequence[0] * 255)
    for frame, image in enumerate(frame_sequence[1:], 1):
        image = np.uint8(image * 255)
        h, w = image.shape

        points, st, err = cv2.calcOpticalFlowPyrLK(prev_image, image, points, None, maxLevel=2,
                                                   winSize=(2 * NEIGHBOUR_DIST + 1, 2 * NEIGHBOUR_DIST + 1))
        alive_mask = st.ravel() == 1
        points = points[alive_mask]
        track_ids = track_ids[alive_mask]
        for track_id, point in zip(track_ids, points):
            tracks[track_id].append(point)
        covered_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        covered_mask[_points_coordinates(points, h, w)] = 1
        covered_mask = cv2.dilate(covered_mask, COVER_KERNEL)
        new_points = cv2.goodFeaturesToTrack(image, CORNERS_PER_TIME, CORNER_QUALITY, NEIGHBOUR_DIST)
        new_points = new_points[covered_mask[_points_coordinates(new_points, h, w)] == 0]
        points = np.concatenate([points, new_points])
        new_ids = np.arange(len(tracks), len(tracks) + len(new_points))
        tracks += [[new_point] for new_point in new_points]
        track_ids = np.concatenate([track_ids, new_ids])
        prev_image = image.copy()
        corners_coordinates = np.squeeze(np.int32(points))
        corner_ids = np.arange(len(corners_coordinates))
        corner_sizes = np.zeros(len(corners_coordinates), dtype=np.int32) + highlight_size
        corners = FrameCorners(corner_ids, corners_coordinates, corner_sizes)
        builder.set_corners_at_frame(frame, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
