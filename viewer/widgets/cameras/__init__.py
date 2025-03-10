import numpy as np
from .. import Widget
from ...types import ViewerMode, Texture2D

# Coordinate system is same as OpenCV
# Forward -> +Z
# Up -> -Y
# Right -> +X
class Camera(Widget):
    def __init__(self, mode: ViewerMode, fov_y: float=30.0, to_world: np.ndarray=None):
        super().__init__(mode)

        # Extrinsics
        self.origin = np.asarray([0.0, 0.0, 0.0])
        self.forward = np.asarray([0.0, 0.0, 1.0])
        self.up = np.asarray([0.0, -1.0, 0.0])
        self.right = np.asarray([1.0, 0.0, 0.0])

        self.last_frame_time = 0
        self.delta_time = 0

        if to_world is not None:
            self.update_pose(to_world)

        self.fov_y = fov_y

    def server_recv(self, _, text):
        self.update_pose(np.array(text["to_world"]))

    def client_send(self):
        return None, self.to_json()

    @classmethod
    def from_json(cls, json):
        to_world = np.array(json["to_world"])
        return cls(to_world)

    def to_json(self):
        return { "to_world": self.to_world.tolist() }

    def process_input(self):
        """ Child class should override this to navigate. """
        pass

    @property
    def to_world(self) -> np.ndarray:
        mat = np.identity(4)
        mat[:3, 3] = self.origin
        mat[:3, 0] = self.right
        mat[:3, 1] = -self.up
        mat[:3, 2] = self.forward
        return mat

    @property
    def to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.to_world)

    def show_gui(self) -> bool:
        return False
    
    def draw_camera(self, camera: 'Camera', texture: Texture2D, thickness: float=1.0, color: tuple=(1.0, 1.0, 1.0)):
        """
        Draw the camera onto texture as observed from another camera. The camera
        is drawn on a OpenGL texture with a fragment shader.

        Args:
            camera: The camera from which the current camera is observed.
            texture: The texture on which the camera is to be drawn.
            thickness: The thickness of the lines in pixels.
            color: The color (normalized) of the camera lines.
        """
        raise NotImplementedError()

    def apply_rotation(self, angle_forward: float, angle_right: float, angle_up: float):
        """
        Rotate the camera about its local axes (forward, right, up).
        Angles are in radians.
        """

        def rotate_vec(vec, axis, angle):
            """Rotate vector `vec` around normalized `axis` by `angle` (radians)."""
            axis = axis / np.linalg.norm(axis)
            c = np.cos(angle)
            s = np.sin(angle)
            dot = np.dot(axis, vec)
            cross = np.cross(axis, vec)
            return c * vec + s * cross + (1 - c) * dot * axis

        if abs(angle_forward) > 1e-7:
            self.up = rotate_vec(self.up, self.forward, angle_forward)
            self.right = rotate_vec(self.right, self.forward, angle_forward)
        if abs(angle_right) > 1e-7:
            self.forward = rotate_vec(self.forward, self.right, angle_right)
            self.up = rotate_vec(self.up, self.right, angle_right)
        if abs(angle_up) > 1e-7:
            self.forward = rotate_vec(self.forward, self.up, angle_up)
            self.right = rotate_vec(self.right, self.up, angle_up)

        # Re-orthonormalize (to handle floating-point drift)
        self.forward /= np.linalg.norm(self.forward)
        # Recompute right as cross of forward with global -Y or some logic:
        # But typically you'd just do cross of (forward, up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        # Recompute up as cross of (right, forward)
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)

    def update_pose(self, mat: np.ndarray):
        self.origin = mat[:3, 3]
        self.forward = mat[:3, 2]
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.up = -mat[:3, 1]
        self.up = self.forward / np.linalg.norm(self.up)
        self.right = mat[:3, 0]
        self.right = self.forward / np.linalg.norm(self.right)

from .fps import FPSCamera