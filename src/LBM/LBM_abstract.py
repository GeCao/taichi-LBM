import taichi as ti
from abc import ABC, abstractmethod

from typing import List, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBMAbstract(ABC):
    def __init__(self, lbm_manager: "LBMManager"):
        self._lbm_manager = lbm_manager
        self._resolution = self._lbm_manager._resolution
        self._dim = self._lbm_manager._dim
        self._vort_shape = self._resolution if self._dim == 2 else (*self._resolution, 3)
        self._Q = self._lbm_manager._Q

        self._method = self._lbm_manager._method
        self._resolution = self._lbm_manager._resolution
        self._dt = self._lbm_manager._dt
        self._dx = self._lbm_manager._dx
        self._tau = self._lbm_manager._tau
        self._c = self._lbm_manager._c
        self._cs2 = self._lbm_manager._cs2
        self._Re = self._lbm_manager._Re
        self._visc = self._lbm_manager._visc
        self._Lmax = self._lbm_manager._Lmax
        self._Vmax = self._Re * self._visc / self._Lmax
        self._Ma = self._visc / self._cs2
        self._total_steps = self._lbm_manager._total_steps

        self._case_parameters = self._lbm_manager._case_parameters

        # 0: fluid, 1: obs, 2: D----, 3. Von-Neumann
        self.m_cell_type = ti.field(dtype=ti.i32, shape=self._resolution)
        self.m_rho = ti.field(dtype=ti.f32, shape=self._resolution)
        self.m_velocity = ti.field(dtype=ti.f32, shape=(*self._resolution, self._dim))
        self.m_vort = ti.field(dtype=ti.f32, shape=self._vort_shape)
        self.m_f = ti.field(dtype=ti.f32, shape=(*self._resolution, self._Q))
        self.m_feq = ti.field(dtype=ti.f32, shape=(*self._resolution, self._Q))
        self.m_pressure = ti.field(dtype=ti.f32, shape=self._resolution)
        self.m_force = ti.field(dtype=ti.f32, shape=(*self._resolution, self._dim))
        self.m_image = ti.field(dtype=ti.f32, shape=(self._resolution[0], self._resolution[1], 3))

        self.m_time = ti.field(dtype=ti.f32, shape=())
        self.m_case = ti.field(dtype=ti.f32, shape=())
        self.m_case[None] = self._lbm_manager._case

    @abstractmethod
    @ti.func
    def initialization(self):
        ...

    @abstractmethod
    @ti.func
    def run_step(self):
        ...

    @abstractmethod
    @ti.func
    def streaming(self):
        ...

    @abstractmethod
    @ti.func
    def cal_feq(self):
        ...

    @abstractmethod
    @ti.func
    def collision(self):
        ...

    @abstractmethod
    @ti.func
    def macro_variable_process(self):
        ...

    @abstractmethod
    @ti.func
    def cal_forces(self):
        ...

    @abstractmethod
    @ti.func
    def compute_image(self):
        ...

    @abstractmethod
    @ti.func
    def compute_vort(self):
        ...

    @abstractmethod
    @ti.kernel
    def kill(self):
        ...
