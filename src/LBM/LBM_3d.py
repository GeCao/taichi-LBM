import numpy as np
import taichi as ti
from .LBM_abstract import LBMAbstract
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_3D(LBMAbstract):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_3D, self).__init__(lbm_manager=lbm_manager)
        self._e = ti.field(dtype=ti.i32, shape=(self._Q, 3))
        self._w = ti.field(dtype=ti.f32, shape=(self._Q,))

        np_e = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                         [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0],
                         [0.0, 0.0, -1.0],
                         [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, -1.0], [0.0, -1.0, -1.0]], dtype=np.float)
        self._e.from_numpy(np_e)

        np_w = np.array([1.0 / 3.0,
                         1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
                         1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                         1.0 / 18.0,
                         1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                         1.0 / 18.0,
                         1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float)
        self._w.from_numpy(np_w)

    @ti.func
    def initialization(self):
        pass

    @ti.func
    def run_step(self):
        self.streaming()
        self.macro_variable_process()
        self.cal_forces()
        self.rebounce_f()
        self.collision()

        self.m_time[None] += self._dt
        self.compute_image()
        # self.error_check()

    @ti.func
    def streaming(self):
        wid, hei, dep = self.m_rho.shape
        for i, j, r in self.m_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0

            self.m_feq[i, j, r, 0] = self.m_f[i, j, r, 0]

            self.m_feq[i_right, j, r, 1] = self.m_f[i, j, r, 1]
            self.m_feq[i, j_behind, r, 2] = self.m_f[i, j, r, 2]
            self.m_feq[i_left, j, r, 3] = self.m_f[i, j, r, 3]
            self.m_feq[i, j_front, r, 4] = self.m_f[i, j, r, 4]

            self.m_feq[i_right, j_behind, r, 5] = self.m_f[i, j, r, 5]
            self.m_feq[i_left, j_behind, r, 6] = self.m_f[i, j, r, 6]
            self.m_feq[i_left, j_front, r, 7] = self.m_f[i, j, r, 7]
            self.m_feq[i_right, j_front, r, 8] = self.m_f[i, j, r, 8]

            self.m_feq[i, j, r_up, 9] = self.m_f[i, j, r, 9]

            self.m_feq[i_right, j, r_up, 10] = self.m_f[i, j, r, 10]
            self.m_feq[i, j_behind, r_up, 11] = self.m_f[i, j, r, 11]
            self.m_feq[i_left, j, r_up, 12] = self.m_f[i, j, r, 12]
            self.m_feq[i, j_front, r_up, 13] = self.m_f[i, j, r, 13]

            self.m_feq[i, j, r_down, 14] = self.m_f[i, j, r, 14]

            self.m_feq[i_right, j, r_down, 15] = self.m_f[i, j, r, 15]
            self.m_feq[i, j_behind, r_down, 16] = self.m_f[i, j, r, 16]
            self.m_feq[i_left, j, r_down, 17] = self.m_f[i, j, r, 17]
            self.m_feq[i, j_front, r_down, 18] = self.m_f[i, j, r, 18]

        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]

    @ti.func
    def rebounce_f(self):
        for I in ti.grouped(self.m_cell_type):
            if self.m_cell_type[I] == 1:
                self.m_f[I, 1], self.m_f[I, 3] = self.m_f[I, 3], self.m_f[I, 1]
                self.m_f[I, 2], self.m_f[I, 4] = self.m_f[I, 4], self.m_f[I, 2]
                self.m_f[I, 5], self.m_f[I, 7] = self.m_f[I, 7], self.m_f[I, 5]
                self.m_f[I, 6], self.m_f[I, 8] = self.m_f[I, 8], self.m_f[I, 6]

                self.m_f[I, 9], self.m_f[I, 14] = self.m_f[I, 14], self.m_f[I, 9]

                self.m_f[I, 10], self.m_f[I, 17] = self.m_f[I, 17], self.m_f[I, 10]
                self.m_f[I, 11], self.m_f[I, 18] = self.m_f[I, 18], self.m_f[I, 11]
                self.m_f[I, 12], self.m_f[I, 15] = self.m_f[I, 15], self.m_f[I, 12]
                self.m_f[I, 13], self.m_f[I, 16] = self.m_f[I, 16], self.m_f[I, 13]

    @ti.func
    def cal_feq(self, vel, I):
        # u2 = 0.0
        # for rr in range(self._dim):
        #     u2 += self.m_velocity[I, rr] * self.m_velocity[I, rr]
        for k in range(self._Q):
            # eu = 0.0
            # for rr in range(self._dim):
            #     eu += self._e[k, rr] * self.m_velocity[I, rr]
            # self.m_feq[I, k] = self.m_rho[I] * self._w[k] * (
            #         1.0 + eu / self._cs2 + 0.5 * eu * eu / self._cs2 / self._cs2 - 0.5 * u2 / self._cs2)
            self.m_feq[I, k] = self.m_rho[I] * self._w[k]
            for rr in ti.static(range(self._dim)):
                self.m_feq[I, k] *= (2.0 - ti.sqrt(1.0 + 3.0 * vel[rr] * vel[rr]))
                self.m_feq[I, k] *= ti.pow((2.0 * vel[rr] + ti.sqrt(1.0 + 3.0 * vel[rr] * vel[rr])) / (1.0 - vel[rr]),
                                      self._e[k, rr])

    @ti.func
    def collision(self):
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        tau = self._tau
        for I, in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                u_tackle[0] = self.m_velocity[I, 0] + tau * (self.m_force[I, 0]) / self.m_rho[I]
                u_tackle[1] = self.m_velocity[I, 1] + tau * (self.m_force[I, 1]) / self.m_rho[I]
                self.cal_feq(u_tackle, I)
                for k in range(self._Q):
                    self.m_f[I, k] = (1.0 - 1.0 / tau) * self.m_f[I, k] + self.m_feq[I, k] / tau

    @ti.func
    def macro_variable_process(self):
        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                self.m_rho[I] = 0.0
                for rr in ti.static(range(self._dim)):
                    self.m_velocity[I, rr] = 0.0

        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                for k in ti.static(range(self._Q)):
                    self.m_rho[I] += self.m_f[I, k]
                    for rr in ti.static(range(self._dim)):
                        self.m_velocity[I, rr] += self.m_f[I, k] * self._e[k, rr]

        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                for rr in ti.static(range(self._dim)):
                    self.m_velocity[I, rr] *= (self._c / self.m_rho[I])

    @ti.func
    def cal_forces(self):
        pass

    @ti.func
    def compute_vort(self):
        wid, hei, dep = self.m_rho.shape
        for i, j, r in self.m_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0
            self.m_vort[i, j, r, 2] = (self.m_velocity[i, j_behind, r, 0] - self.m_velocity[i, j_front, r, 0]) / (2.0 * self._dx) - \
                                      (self.m_velocity[i_right, j, r, 1] - self.m_velocity[i_left, j, r, 1]) / (2.0 * self._dx)

            self.m_vort[i, j, r, 1] = (self.m_velocity[i_right, j, r, 2] - self.m_velocity[i_left, j, r, 2]) / (2.0 * self._dx) - \
                                      (self.m_velocity[i, j, r_up, 0] - self.m_velocity[i, j, r_down, 0]) / (2.0 * self._dx)

            self.m_vort[i, j, r, 0] = (self.m_velocity[i, j, r_up, 1] - self.m_velocity[i, j, r_down, 1]) / (2.0 * self._dx) - \
                                      (self.m_velocity[i, j_behind, r, 2] - self.m_velocity[i, j_front, r, 2]) / (2.0 * self._dx)

    @ti.func
    def compute_image(self):
        image_is_vort = self._case_parameters[self._lbm_manager._case]['image'] == 'vort'
        image_is_rho = self._case_parameters[self._lbm_manager._case]['image'] == 'rho'
        image_is_velocity = self._case_parameters[self._lbm_manager._case]['image'] == 'velocity'
        if image_is_vort:
            self.compute_vort()
        for i, j, rgb in self.m_image:
            if image_is_vort:
                pass
            elif image_is_rho:
                self.m_image[i, j, rgb] = self.m_rho[i, j, self._resolution[2] // 2] * 1.0
            elif image_is_velocity:
                pass

    @ti.func
    def error_check(self):
        pass

    @ti.kernel
    def kill(self):
        pass
