import numpy as np
from . import Camera
from ...types import ViewerMode
from imgui_bundle import imgui

# TODO: Coalesce all camera types into a single class
class FPSCamera(Camera):
    def __init__(
            self, mode: ViewerMode,
            res_x: int=1280, res_y: int=720, fov_y: float=30.0,
            z_near: float=0.001, z_far: float=100.0,
            to_world: np.ndarray=None
    ):
        super().__init__(mode, res_x, res_y, fov_y, z_near, z_far, to_world)
        self.speed = 1
        self.mouse_speed = 2
        self.radians_per_pixel = np.pi / 150
        self.invert_mouse = False
    
    def process_mouse_input(self) -> bool:
        if imgui.is_mouse_dragging(0):
            delta = imgui.get_mouse_drag_delta()
            delta.y *= -1 if self.invert_mouse else 1
            delta.x *= -1 if self.invert_mouse else 1
            angle_right = -delta.y * self.radians_per_pixel * self.delta_time * self.mouse_speed
            angle_up = -delta.x * self.radians_per_pixel * self.delta_time * self.mouse_speed
            self.apply_rotation(0, angle_right, angle_up)
            imgui.reset_mouse_drag_delta()
            return True

        return False
    
    def process_keyboard_input(self):
        update = False

        if imgui.is_key_down(imgui.Key.w):
            self.origin += self.speed * self.forward * self.delta_time
            update = True
        if imgui.is_key_down(imgui.Key.a):
            self.origin -= self.speed * self.right * self.delta_time
            update = True
        if imgui.is_key_down(imgui.Key.q):
            self.origin -= self.speed * self.up * self.delta_time
            update = True
        if imgui.is_key_down(imgui.Key.s):
            self.origin -= self.speed * self.forward * self.delta_time
            update = True
        if imgui.is_key_down(imgui.Key.d):
            self.origin += self.speed * self.right * self.delta_time
            update = True
        if imgui.is_key_down(imgui.Key.e):
            self.origin += self.speed * self.up * self.delta_time
            update = True
        
        angle_forward = 0.0
        angle_right = 0.0
        angle_up = 0.0
        if imgui.is_key_down(imgui.Key.i):
            angle_right += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(imgui.Key.k):
            angle_right -= 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(imgui.Key.j):
            angle_up += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(imgui.Key.l):
            angle_up -= 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(imgui.Key.o):
            angle_forward += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(imgui.Key.u):
            angle_forward -= 50 * self.radians_per_pixel * self.delta_time

        if angle_forward or angle_right or angle_up:
            self.apply_rotation(angle_forward, angle_right, angle_up)
            update = True

        return update
    
    def show_gui(self):
        # TODO: Properly do this (respect focus etc)
        super().show_gui()
        imgui.checkbox("Invert Mouse", self.invert_mouse)