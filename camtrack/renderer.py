#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


FRUSTUM_LEN = 12
FRUSTUM_COLOR = np.array([[1, 1, 0]])
NEAR = 0.01
FAR = 100


def _build_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 vp;
        uniform mat4 m;

        attribute vec3 position;
        attribute vec3 color;
        attribute int mask;
        varying vec3 f_color;
        
        void main() {
            if (mask != 0) {
                gl_Position =  vp * m * vec4(position, 1.0);
            } else {
                gl_Position = vp * vec4(position, 1.0);
            }
            f_color = color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        varying vec3 f_color;
        out vec3 out_color;

        void main() {
            out_color = f_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )
    return shaders.compileProgram(vertex_shader, fragment_shader)


def calc_fy(camera_fov_y):
    return 2 * np.tan(camera_fov_y / 2)


def _make_frustum(camera_fov_y, aspect_ratio):
    fy =  calc_fy(camera_fov_y)
    fx = fy / aspect_ratio
    return np.array([
        [0, 0, 0],
        [-fx, -fy, -1],
        [0, 0, 0],
        [-fx, fy, -1],
        [0, 0, 0],
        [fx, -fy, -1],
        [0, 0, 0],
        [fx, fy, -1],
        [-fx, -fy, -1],
        [-fx, fy, -1],
        [fx, fy, -1],
        [fx, -fy, -1]
    ])


def _make_m(camera_tr_vec, camera_rot_mat):
    return np.block([
        [np.linalg.inv(camera_rot_mat).T, np.expand_dims(camera_tr_vec, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))]
    ])


def _make_vp(camera_tr_vec, camera_rot_mat, camera_fov_y):
    v = np.linalg.inv(np.block([
        [camera_rot_mat, camera_tr_vec.reshape((3, 1))],
        [np.zeros((1, 3)), np.ones((1, 1))]
    ]))

    aspect_ratio = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
    fy = calc_fy(camera_fov_y)
    fx = fy / aspect_ratio
    # camera_fov_x = camera_fov_y / aspect_ratio
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = fx
    p[1, 1] = fy
    p[2, 2] = -(FAR + NEAR) / (FAR - NEAR)
    p[2, 3] = -2 * FAR * NEAR / (FAR - NEAR)
    p[3, 2] = -1

    return p @ v


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self.cam_positions = np.array([i.t_vec for i in tracked_cam_track])
        self.cam_rotations = np.array([i.r_mat for i in tracked_cam_track])

        self.n_points = len(point_cloud.points)
        self.track_len = len(tracked_cam_track)
        self.point_coordinates = np.concatenate([
            point_cloud.points * np.array([[1, -1, -1]]),
            self.cam_positions,
            _make_frustum(tracked_cam_parameters.fov_y, tracked_cam_parameters.aspect_ratio)
        ]).astype(np.float32)
        self.point_colors = np.concatenate([
            point_cloud.colors,
            np.ones_like(self.cam_positions),
            np.repeat(FRUSTUM_COLOR, FRUSTUM_LEN, axis=0)
        ]).astype(np.float32)
        self.mask = np.concatenate([
            np.zeros(self.n_points + self.track_len, dtype=np.int32),
            np.ones(FRUSTUM_LEN, dtype=np.int32)
        ])

        self.color_buffer = vbo.VBO(self.point_colors)
        self.coordinates_buffer = vbo.VBO(self.point_coordinates)
        self.mask_buffer = vbo.VBO(self.mask)

        self.program = _build_program()
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

        self.color_buffer.bind()
        col_loc = GL.glGetAttribLocation(self.program, 'color')
        GL.glEnableVertexAttribArray(col_loc)
        GL.glVertexAttribPointer(col_loc, 3, GL.GL_FLOAT, False, 0, self.color_buffer)
        self.color_buffer.unbind()

        self.mask_buffer.bind()
        mask_loc = GL.glGetAttribLocation(self.program, 'mask')
        GL.glEnableVertexAttribArray(mask_loc)
        GL.glVertexAttribPointer(mask_loc, 1, GL.GL_INT, False, 0, self.mask_buffer)
        self.mask_buffer.unbind()

        self.coordinates_buffer.bind()
        position_loc = GL.glGetAttribLocation(self.program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT, False, 0, self.coordinates_buffer)
        self.coordinates_buffer.unbind()

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        m = _make_m(self.cam_positions[tracked_cam_track_pos], self.cam_rotations[tracked_cam_track_pos])
        vp = _make_vp(camera_tr_vec, camera_rot_mat, camera_fov_y)
        self._render_points(m, vp)
        GLUT.glutSwapBuffers()

    def _render_points(self, m, vp):
        shaders.glUseProgram(self.program)

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, 'm'), 1, True, m)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, 'vp'), 1, True, vp)

        GL.glDrawArrays(GL.GL_LINE_STRIP, self.n_points, self.track_len)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.n_points)
        GL.glDrawArrays(GL.GL_LINES, self.n_points + self.track_len, 8)
        GL.glDrawArrays(GL.GL_LINE_LOOP, self.n_points + self.track_len + 8, 4)
        shaders.glUseProgram(0)
