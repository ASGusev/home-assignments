from typing import List

import numpy as np
import scipy as sp
import sortednp as snp
import scipy.sparse
from scipy.optimize import approx_fprime
import cv2

from corners import FrameCorners
from _camtrack import PointCloudBuilder, rodrigues_and_translation_to_view_mat3x4, calc_inlier_indices, \
    compute_reprojection_errors


EPS = 1e-8
FRAMES_STEP = 200
ALPHA_INITIAL = 8
ALPHA_DEC_STEP = 2 ** .5
ALPHA_INC_STEP = 8
ITERATIONS = 5


def derivative_at_rtp_d(r, t, point3d, point2d, intrinsic_mat):
    point2d = np.concatenate((point2d, np.ones(1)))

    def f(rtp):
        r_vec = rtp[:3].reshape(3, 1)
        t_vec = rtp[3:6].reshape(3, 1)
        p = rtp[6:]
        proj = intrinsic_mat @ rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec) @ p
        proj /= proj[2]
        return np.square(proj - point2d).sum()

    xk = np.concatenate((r, t, point3d, np.ones(1)))
    derivative = approx_fprime(xk, f, EPS)
    return derivative[:3], derivative[3:6], derivative[6:9]


def get_inliers(intrinsic_mat: np.ndarray,
                list_of_corners: List[FrameCorners],
                max_inlier_reprojection_error: float,
                view_mats: List[np.ndarray],
                pc_builder: PointCloudBuilder):
    inliers = []
    for view_mat, corners in zip(view_mats, list_of_corners):
        common_ids, (corner_ids, cloud_ids) = snp.intersect(
            corners.ids.flatten(), pc_builder.ids.flatten(), indices=True)
        points3d = pc_builder.points[cloud_ids]
        points2d = corners.points[corner_ids]
        inlier_indices = calc_inlier_indices(
            points3d, points2d, intrinsic_mat @ view_mat, max_inlier_reprojection_error)
        inliers.append((common_ids[inlier_indices], corner_ids[inlier_indices]))
    return inliers


def get_matches(inliers, n_frames, start_index):
    matches = [
        (inlier_id, frame_id, inlier_index)
        for frame_id, (frame_inliers, inlier_indices) in enumerate(inliers, start_index)
        for inlier_id, inlier_index in zip(frame_inliers, inlier_indices)
    ]
    matches.sort(key=lambda x: x[0] * n_frames + x[1])
    return matches


def block_inv(v):
    v_inv = sp.sparse.lil_matrix(v.shape)
    for j in range(v.shape[0] // 3):
        v_inv[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3] = \
            np.linalg.inv(v[j * 3:(j + 1) * 3, j * 3:(j + 1) * 3].toarray())
    return v_inv


def calc_jacobian(matches, cameras_params_dim, rs, ts, pc_builder, list_of_corners, point_id_to_cur,
                  point_id_to_cloud_pos, intrinsic_mat, start_index):
    jacobian = sp.sparse.lil_matrix((len(matches), cameras_params_dim + 3 * len(point_id_to_cur)))
    for i, (point_id, frame_id, point_index) in enumerate(matches):
        dr, dt, dp = derivative_at_rtp_d(
            rs[frame_id], ts[frame_id], pc_builder.points[point_id_to_cloud_pos[point_id]],
            list_of_corners[frame_id].points[point_index], intrinsic_mat)
        frame_id -= start_index
        jacobian[i, 6 * frame_id: 6 * frame_id + 3] = dr
        jacobian[i, 6 * frame_id + 3: 6 * frame_id + 6] = dt
        point_cur_ind = point_id_to_cur[point_id]
        jacobian[i, cameras_params_dim + point_cur_ind * 3:cameras_params_dim + (point_cur_ind + 1) * 3] = dp
    return jacobian


def get_residuals(pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, matches, view_mats):
    return np.square([
        compute_reprojection_errors(
            pc_builder.points[point_id_to_cloud[point_id]:point_id_to_cloud[point_id] + 1],
            list_of_corners[frame_id].points[point_index:point_index + 1], intrinsic_mat @ view_mats[frame_id]
        )[0]
        for point_id, frame_id, point_index in matches
    ])


def calc_reprojection_error(inliers, pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, view_mats):
    matches = get_matches(inliers, len(list_of_corners), 0)
    us = get_residuals(pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, matches, view_mats)
    return us.mean()


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    n_frames = len(view_mats)
    inliers = get_inliers(intrinsic_mat, list_of_corners, max_inlier_reprojection_error, view_mats, pc_builder)
    rs = [cv2.Rodrigues(i[:, :3])[0].flatten() for i in view_mats]
    ts = [i[:, 3] for i in view_mats]
    point_id_to_cloud = -np.ones(pc_builder.ids.max() + 1, dtype=np.int32)
    point_id_to_cloud[pc_builder.ids.flatten()] = np.arange(len(pc_builder.ids))

    re = calc_reprojection_error(inliers, pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, view_mats)
    print('Reprojection error before bundle adjustment:', re)
    for start in range(0, n_frames, FRAMES_STEP):
        end = min(start + FRAMES_STEP, n_frames)
        cameras_params_dim = (end - start) * 6

        matches = get_matches(inliers[start:end], len(list_of_corners), start)
        relevant_point_ids = np.array(list(sorted({point_id for point_id, frame_id, point_index in matches})))
        point_id_to_cur = {p: i for i, p in enumerate(relevant_point_ids)}
        _, (_, cloud_indices) = snp.intersect(relevant_point_ids, pc_builder.ids.flatten(), indices=True)
        relevant_points = pc_builder.points[cloud_indices]

        alpha = ALPHA_INITIAL
        us_mean_prev = -float('inf')

        for i in range(ITERATIONS):
            us = get_residuals(pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, matches, view_mats)
            jacobian = calc_jacobian(matches, cameras_params_dim, rs, ts, pc_builder, list_of_corners,
                                     point_id_to_cur, point_id_to_cloud, intrinsic_mat, start)
            jtj = jacobian.T @ jacobian

            us_mean = us.mean()
            alpha = alpha / ALPHA_DEC_STEP if us_mean < us_mean_prev else alpha * ALPHA_INC_STEP
            us_mean_prev = us_mean
            alpha_multiplier = 1 + alpha
            jtj[np.arange(jtj.shape[0]), np.arange(jtj.shape[0])] *= alpha_multiplier

            u = jtj[:cameras_params_dim, :cameras_params_dim]
            v = jtj[cameras_params_dim:, cameras_params_dim:]
            w = jtj[:cameras_params_dim, cameras_params_dim:]
            wt = jtj[cameras_params_dim:, :cameras_params_dim]
            v_inv = block_inv(v)
            g = jacobian.T @ us
            b = w @ v_inv @ g[cameras_params_dim:] - g[:cameras_params_dim]
            a = (u - w @ v_inv @ wt).toarray()
            delta_c = np.linalg.solve(a, b)
            delta_x = v_inv @ (-g[cameras_params_dim:] - wt @ delta_c)
            for k, j in enumerate(range(start, end)):
                rs[j] += delta_c[k * 6: k * 6 + 3]
                ts[j] += delta_c[k * 6 + 3: k * 6 + 6]
                view_mats[j] = rodrigues_and_translation_to_view_mat3x4(rs[j], ts[j].reshape(3, 1))
            relevant_points += delta_x.reshape((-1, 3))
            pc_builder.update_points(relevant_point_ids, relevant_points)
    re = calc_reprojection_error(inliers, pc_builder, point_id_to_cloud, list_of_corners, intrinsic_mat, view_mats)
    print('Reprojection error after bundle adjustment:', re)
    return view_mats
