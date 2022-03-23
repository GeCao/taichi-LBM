import taichi as ti
from .LBM_3d import LBM_3D
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_3D_SC(LBM_3D):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_3D_SC, self).__init__(lbm_manager=lbm_manager)
        self.m_psx = ti.field(ti.f32, shape=self._resolution)
        self._rho_wall = self._case_parameters[self._lbm_manager._case]['rho_wall']
        self._rho_liquid = self._case_parameters[self._lbm_manager._case]['rho_liquid']
        self._rho_gas = self._case_parameters[self._lbm_manager._case]['rho_gas']

    @ti.func
    def initialization(self):
        for i, j, r in self.m_rho:
            if ti.abs(i - self._resolution[0] * 0.5) < 0.25 * self._resolution[0] and \
                    ti.abs(j - self._resolution[1] * 0.5) < 0.25 * self._resolution[1] and \
                    ti.abs(r - self._resolution[2] * 0.5) < 0.25 * self._resolution[2]:
                self.m_rho[i, j, r] = self._rho_liquid
            else:
                self.m_rho[i, j, r] = self._rho_gas
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        for I in ti.grouped(self.m_rho):
            for rr in ti.static(range(self._dim)):
                u_tackle[rr] = self.m_velocity[I, rr]
            self.cal_feq(u_tackle, I)
        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]

    @ti.func
    def cal_forces(self):
        R = 1.0
        a = 12 * self._cs2
        b = 4.0
        G1 = -1.0 / 3.0
        Tc = 0.3773 * a / (b * R)
        TT0 = 0.875
        TT = TT0 * Tc

        TT = self._cs2 / R
        for I in ti.grouped(self.m_psx):
            if self.m_cell_type[I] == 0:
                temp_rho = self.m_rho[I] * b / 4.0
                temp_Fxy = R * TT * (1.0 + (4.0 * temp_rho - 2.0 * temp_rho**2) / ti.pow((1.0 - temp_rho), 3)) - \
                           a * temp_rho - self._cs2

                self.m_psx[I] = ti.sqrt(2.0 * self.m_rho[I] * temp_Fxy / G1 / self._cs2)
                self.m_pressure[I] = self.m_rho[I] * self._cs2 + 0.5 * self._cs2 * G1 * self.m_psx[I] * self.m_psx[I]

        temp_rho = self._rho_wall * b / 4.0
        temp_Fxy = R * TT * (1.0 + (4.0 * temp_rho - 2.0 * temp_rho**2) / ti.pow((1.0 - temp_rho), 3)) - \
                   a * temp_rho - self._cs2
        psx_w = ti.sqrt(2.0 * self._rho_wall * temp_Fxy / G1 / self._cs2)

        G1 = -1.0 / 3.0
        for i, j, r in self.m_psx:
            for rr in ti.static(range(self._dim)):
                self.m_force[i, j, r, rr] = 0.0
            if self.m_cell_type[i, j, r] == 0:
                for k in range(1, self._Q):
                    xp = i + self._e[k, 0]
                    yp = j + self._e[k, 1]
                    zp = r + self._e[k, 2]
                    if xp < 0:
                        xp = self._resolution[0] - 1
                    elif xp >= self._resolution[0]:
                        xp = 0
                    if yp < 0:
                        yp = self._resolution[1] - 1
                    elif yp >= self._resolution[1]:
                        yp = 0
                    if zp < 0:
                        zp = self._resolution[2] - 1
                    elif zp >= self._resolution[2]:
                        zp = 0
                    for rr in ti.static(range(self._dim)):
                        if self.m_cell_type[xp, yp, zp] == 1:
                            self.m_force[i, j, r, rr] += self._w[k] * self._e[k, rr] * psx_w
                        else:
                            self.m_force[i, j, r, rr] += self._w[k] * self._e[k, rr] * self.m_psx[xp, yp, zp]
                for rr in ti.static(range(self._dim)):
                    self.m_force[i, j, r, rr] *= (-G1 * self.m_psx[i, j, r] * self._c)