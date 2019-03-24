#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import cv2
import sortednp as snp

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


TRIANGULATION_PARAMETERS = TriangulationParameters(1, 8, .1)
ADDITIONAL_TRIANGULATION_PARAMETERS = TriangulationParameters(1, 7, 1)
MAX_TRIANGULATION_COS = .98
MAX_TRIANGULATION_PARTNERS = 6
MAX_PROJECTION_ERROR = 20


def try_initialize(first_corners: FrameCorners, other_corners: FrameCorners, intrinsic_mat):
    correspondences = build_correspondences(first_corners, other_corners)
    essential_matrix, _ = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, intrinsic_mat)
    r1, r2, t = cv2.decomposeEssentialMat(essential_matrix)
    max_points = -float('inf')
    res_points3d, res_ids = None, None
    for r in r1, r2:
        for i in t, -t:
            view_matrix = np.hstack((r, i.reshape(3, 1)))
            points3d, ids = triangulate_correspondences(
                correspondences, eye3x4(), view_matrix, intrinsic_mat, TRIANGULATION_PARAMETERS)
            if len(points3d) > max_points:
                max_points = len(points3d)
                res_points3d, res_ids = points3d, ids
    return res_points3d, res_ids


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    best_len = -1
    best_points3d, best_ids = None, None
    for i in range(1, len(corner_storage)):
        points3d, ids = try_initialize(corner_storage[0], corner_storage[i], intrinsic_mat)
        if len(points3d) > best_len:
            best_points3d, best_ids = points3d, ids
            best_len = len(points3d)
    cloud_builder = PointCloudBuilder(best_ids, best_points3d)
    camera_positions = [np.hstack([np.eye(3), np.zeros((3, 1))])]
    prev_t = np.zeros((3, 1))
    prev_r = np.array([[0], [0], [1]])
    rot_vectors = np.array([[0, 0, 1]])
    bad = set()
    for frame_corners in corner_storage[1:]:
        common_ids, (corner_ids, cloud_ids) = snp.intersect(
            frame_corners.ids.flatten(), cloud_builder.ids.flatten(), indices=True)
        not_bad = np.array([i not in bad for i in common_ids])
        points3d = cloud_builder.points[cloud_ids[not_bad]]
        points2d = frame_corners.points[corner_ids[not_bad]].astype(np.float32)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(points3d, points2d, intrinsic_mat, None, confidence=.999,
                                                         reprojectionError=10, iterationsCount=250, tvec=prev_t,
                                                         rvec=prev_r, useExtrinsicGuess=True)
        prev_r, prev_t = rvec, tvec
        rvec = rvec.get()
        tvec = tvec.get()
        cur_view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        projection_errors = compute_reprojection_errors(points3d, points2d, intrinsic_mat @ cur_view_mat)
        bad.update(common_ids[not_bad][projection_errors > MAX_PROJECTION_ERROR])
        camera_positions.append(cur_view_mat)
        rvec /= np.linalg.norm(rvec)
        coss = rot_vectors @ rvec
        i = len(rot_vectors) - 1
        triangulation_partners = []
        while i >= 0 and len(triangulation_partners) < MAX_TRIANGULATION_PARTNERS:
            if coss[i] <= MAX_TRIANGULATION_COS:
                triangulation_partners.append(i)
            i -= 1
        for other in triangulation_partners:
            common_ids, (indices_here, indices_other) = snp.intersect(
                frame_corners.ids.flatten(), corner_storage[other].ids.flatten(), indices=True)
            used_ids = set(cloud_builder.ids.flatten())
            unused_ids = np.array([i not in used_ids for i in common_ids])
            if np.all(np.logical_not(unused_ids)):
                continue
            cur_correspondences = Correspondences(
                common_ids[unused_ids], np.float32(frame_corners.points[indices_here[unused_ids]]),
                np.float32(corner_storage[other].points[indices_other[unused_ids]]))
            new_points, new_ids = triangulate_correspondences(
                cur_correspondences, cur_view_mat, camera_positions[other], intrinsic_mat,
                ADDITIONAL_TRIANGULATION_PARAMETERS)
            cloud_builder.add_points(new_ids, new_points)
        rot_vectors = np.vstack((rot_vectors, rvec.T))
    return camera_positions, cloud_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
