import taichi as ti
from .LBM_2d import LBM_2D
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_2D_KBC(LBM_2D):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_2D_KBC, self).__init__(lbm_manager=lbm_manager)

        self.C_matrix = ti.field(ti.f32, (3, 3, self._Q))

        np_C_matrix = np.zeros((3, 3, self._Q), dtype=np.float)
        np_C_matrix[0, 0, :] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        np_C_matrix[1, 0, :] = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        np_C_matrix[0, 1, :] = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        np_C_matrix[2, 0, :] = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1])
        np_C_matrix[0, 2, :] = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1])
        np_C_matrix[1, 1, :] = np.array([0, 0, 0, 0, 0, 1, -1, 1, -1])
        np_C_matrix[2, 2, :] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
        np_C_matrix[2, 1, :] = np.array([0, 0, 0, 0, 0, 1, 1, -1, -1])
        np_C_matrix[1, 2, :] = np.array([0, 0, 0, 0, 0, 1, -1, -1, 1])
        self.C_matrix.from_numpy(np_C_matrix)

    @ti.func
    def calculate_KBC_params(self, I):
        moment = ti.Matrix.zero(ti.f32, 3, 3)
        moment_eq = ti.Matrix.zero(ti.f32, 3, 3)
        KBC_delta_s = ti.Vector.zero(ti.f32, self._Q)
        KBC_delta_h = ti.Vector.zero(ti.f32, self._Q)
        for k in range(self._Q):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    moment[i, j] += self.m_f[I, k] * self.C_matrix[i, j, k]
                    moment_eq[i, j] += self.m_feq[I, k] * self.C_matrix[i, j, k]

        # KBC - A
        KBC_T = moment[2, 0] + moment[0, 2]
        KBC_N = moment[2, 0] - moment[0, 2]
        KBC_PIxy = moment[1, 1]
        KBC_Qxxy, KBC_Qxyy = moment[2, 1], moment[1, 2]
        KBC_A = moment[2, 2]

        KBC_delta_s[0] = (-KBC_T)

        KBC_delta_s[1] = 0.5 * (0.5 * (KBC_T + KBC_N))
        KBC_delta_s[3] = 0.5 * (0.5 * (KBC_T + KBC_N))

        KBC_delta_s[2] = 0.5 * (0.5 * (KBC_T - KBC_N))
        KBC_delta_s[4] = 0.5 * (0.5 * (KBC_T - KBC_N))

        KBC_delta_s[5] = 0.25 * (KBC_PIxy)
        KBC_delta_s[6] = 0.25 * (-KBC_PIxy)
        KBC_delta_s[7] = 0.25 * (KBC_PIxy)
        KBC_delta_s[8] = 0.25 * (-KBC_PIxy)

        KBC_T = moment_eq[2, 0] + moment_eq[0, 2]
        KBC_N = moment_eq[2, 0] - moment_eq[0, 2]
        KBC_PIxy = moment_eq[1, 1]
        KBC_Qxxy, KBC_Qxyy = moment_eq[2, 1], moment_eq[1, 2]
        KBC_A = moment_eq[2, 2]

        KBC_delta_s[0] -= (-KBC_T)

        KBC_delta_s[1] -= 0.5 * (0.5 * (KBC_T + KBC_N))
        KBC_delta_s[3] -= 0.5 * (0.5 * (KBC_T + KBC_N))

        KBC_delta_s[2] -= 0.5 * (0.5 * (KBC_T - KBC_N))
        KBC_delta_s[4] -= 0.5 * (0.5 * (KBC_T - KBC_N))

        KBC_delta_s[5] -= 0.25 * (KBC_PIxy)
        KBC_delta_s[6] -= 0.25 * (-KBC_PIxy)
        KBC_delta_s[7] -= 0.25 * (KBC_PIxy)
        KBC_delta_s[8] -= 0.25 * (-KBC_PIxy)

        gamma_nominator, gamma_denominator = 0.0, 0.0
        KBC_delta_h[0] = self.m_f[I, 0] - self.m_feq[I, 0] - KBC_delta_s[0]
        for k in ti.static(range(1, self._Q)):
            KBC_delta_h[k] = self.m_f[I, k] - self.m_feq[I, k] - KBC_delta_s[k]
            gamma_nominator += KBC_delta_s[k] * KBC_delta_h[k] / (self.m_feq[I, k] + 1e-7)
            gamma_denominator += KBC_delta_h[k] * KBC_delta_h[k] / (self.m_feq[I, k] + 1e-7)
        KBC_gamma = gamma_nominator / (gamma_denominator + 1e-7)

        beta = 0.5 / self._tau

        KBC_gamma = (1.0 - (2.0 * beta - 1.0) * KBC_gamma) / beta
        # KBC_gamma = 2.0

        for k in ti.static(range(self._Q)):
            self.m_f[I, k] = self.m_f[I, k] - \
                             2.0 * beta * KBC_delta_s[k] - \
                             beta * KBC_delta_h[k] * KBC_gamma

    @ti.func
    def collision(self):
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        tau = self._tau
        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                u_tackle[0] = self.m_velocity[I, 0] + tau * (self.m_force[I, 0]) / self.m_rho[I]
                u_tackle[1] = self.m_velocity[I, 1] + tau * (self.m_force[I, 1]) / self.m_rho[I]
                self.cal_feq(u_tackle, I)
                self.calculate_KBC_params(I)
