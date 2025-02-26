import numpy as np
from . import Camera
from imgui_bundle import imgui
import glfw

class FPSCamera(Camera):
    def __init__(self, to_world: np.ndarray=None, name=""):
        self.name = name
        self.speed = 1
        self.mouse_speed = 2
        self.radians_per_pixel = np.pi / 150
        self.invert_mouse = False
        self.window = None
        super().__init__(to_world)

    def process_input(self) -> bool:
        curr_time = imgui.get_time()
        self.delta_time = curr_time - self.last_frame_time
        self.last_frame_time = curr_time

        return self._move() or self._rotate()

    def _move(self) -> bool:
        update = False

        if self.window is not None:
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
                self.origin += self.speed * self.forward * self.delta_time
                update = True
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: 
                self.origin -= self.speed * self.right * self.delta_time
                update = True
            if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                self.origin -= self.speed * self.up * self.delta_time
                update = True
            if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
                self.origin -= self.speed * self.forward * self.delta_time
                update = True
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
                self.origin += self.speed * self.right * self.delta_time
                update = True
            if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                self.origin += self.speed * self.up * self.delta_time
                update = True
            
            angle_forward = 0.0
            angle_right = 0.0
            angle_up = 0.0
            if glfw.get_key(self.window, glfw.KEY_I) == glfw.PRESS:
                angle_right += 50 * self.radians_per_pixel * self.delta_time
            if glfw.get_key(self.window, glfw.KEY_K) == glfw.PRESS:
                angle_right -= 50 * self.radians_per_pixel * self.delta_time
            if glfw.get_key(self.window, glfw.KEY_J) == glfw.PRESS:
                angle_up += 50 * self.radians_per_pixel * self.delta_time
            if glfw.get_key(self.window, glfw.KEY_L) == glfw.PRESS:
                angle_up -= 50 * self.radians_per_pixel * self.delta_time
            if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS:
                angle_forward += 50 * self.radians_per_pixel * self.delta_time
            if glfw.get_key(self.window, glfw.KEY_U) == glfw.PRESS:
                angle_forward -= 50 * self.radians_per_pixel * self.delta_time

        if angle_forward or angle_right or angle_up:
            self.apply_rotation(angle_forward, angle_right, angle_up)
            update = True

        return update
    
    def _rotate(self) -> bool:
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
    
    def show_gui(self):
        _, self.speed = imgui.drag_float(f"Movement Speed{self.name}", self.speed, 0.1, 0, 1e8, format="%.1f")
        _, self.invert_mouse = imgui.checkbox(f"Invert Mouse{self.name}", self.invert_mouse)
        _, self.mouse_speed = imgui.drag_float(f"Mouse Speed{self.name}", self.mouse_speed, 0.1, 0, 1e8, format="%.1f")
        return False