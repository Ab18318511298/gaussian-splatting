import glfw
from websockets.sync.server import serve
from .types import *
from viewer.widgets import Widget
from abc import ABC, abstractmethod
from imgui_bundle import immapp, hello_imgui, glfw_utils

# TODO: Figure out minimize crash
class Viewer(ABC):
    """
    Base class for viewer. This class setups up the relevant ImGui callbacks.
    The child class must override the 'show_gui' function to build the GUI.
    It can can also override the 'step' function to perform any per frame
    computations required.
    The '(send|recv)_(server|client)' should be used for message passing between
    the server and client for remote viewer support.
    """
    window_title = "Viewer"
    should_exit = False

    def setup(self):
        """ Go over all of the widgets and initialize them """
        self.glfw_win = glfw_utils.glfw_window_hello_imgui()
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.setup()

    def destroy(self):
        """ Go over all of the widgets and free any manually allocated objects """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.destroy()
    
    def _show_gui(self):
        """
        TODO: Update
        Internal method which handles inputs, resize and calls
        backend computation and then creates the UI.
        """
        if self.mode is CLIENT:
            self._client_send()
        if self.mode is SERVER:
            self._server_recv()

        self.step()

        if self.mode is SERVER:
            self._server_send()
        if self.mode is CLIENT:
            self._client_recv()
        
        self.show_gui()

    def _server_send(self):
        """
        Internal method which goes over all of the registered widgets to compile 
        and send the server state to the client.
        """
        metadata = {}   # Metadata for each widget (Should be JSON serializable)
        all_binaries = []   # List of all binaries to be sent
        binary_to_widget = []   # Mapping of which binary is for which widget
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                binary, text = widget.server_send()
                if text is not None:
                    metadata[widget.widget_id] = text
                if binary is not None:
                    all_binaries.append(binary)
                    binary_to_widget.append(widget.widget_id)
        
        # Add global state (viewer)
        binary, text = self.server_send()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            all_binaries.append(binary)
            binary_to_widget.append("viewer")
        
        # Send the metadata

        # Send binary mapping

        # Send binaries

    def _server_recv(self):
        """
        Internal method which receives state from the client and updates all of
        the widgets.
        """
        pass

    def _client_send(self):
        """
        Internal method which goes over all of the registered widgets to compile
        and send the client state to the server.
        """
        metadata = {}   # Metadata for each widget (Should be JSON serializable)
        all_binaries = []   # List of all binaries to be sent
        binary_to_widget = []   # Mapping of which binary is for which widget
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                binary, text = widget.client_send()
                if text is not None:
                    metadata[widget.widget_id] = text
                if binary is not None:
                    all_binaries.append(binary)
                    binary_to_widget.append(widget.widget_id)
        
        # Add global state (viewer)
        binary, text = self.client_send()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            all_binaries.append(binary)
            binary_to_widget.append("viewer")
        
        # Send the metadata

        # Send binary mapping

        # Send binaries

    def _server_send(self):
        """
        Internal method which receives state from the server and updates all of
        the widgets.
        """
        pass

    def run(self, mode: ViewerMode, ip: str = "localhost", port: int = 6009):
        self.mode = mode
        self.create_widgets()
        if mode in LOCAL_CLIENT:
            self._runner_params = hello_imgui.RunnerParams()
            self._runner_params.fps_idling.enable_idling = False
            self._runner_params.app_window_params.window_geometry.window_size_state = hello_imgui.WindowSizeState.maximized
            self._runner_params.app_window_params.window_title = self.window_title
            self._runner_params.imgui_window_params.show_status_bar = True
            self._runner_params.imgui_window_params.show_menu_bar = True
            self._runner_params.callbacks.post_init = self.setup
            self._runner_params.callbacks.before_exit = self.destroy
            self._runner_params.callbacks.show_gui = self._show_gui
            self._runner_params.callbacks.show_status = self.show_status
            self._runner_params.callbacks.post_init_add_platform_backend_callbacks = lambda: glfw.swap_interval(0)
            self._runner_params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
            self._addon_params = immapp.AddOnsParams(with_implot=True)

            # This is required to make 'want_capture_*' work. The default value is to create a full screen window,
            # but that would mean the 'want_capture_mouse' variable will always be set.
            self._runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
            immapp.run(self._runner_params, self._addon_params)

            if mode is CLIENT:
                pass
                # Connect to the server
        else:
            # Start the server
            pass

    def step(self):
        """ Your application logic goes here. """
        # TODO: Image calculation stuff here
        pass

    def create_widgets(self):
        """ Define stateful widgets here. """
    
    def server_send(self):
        """ Send global viewer state to the client. """
        return None, None
    
    def server_recv(self):
        """ Receive and process global viewer state from the client. """

    def client_send(self):
        """ Send global viewer state to the server. """
        return None, None

    def client_recv(self):
        """ Receive and process global viewer state from the server. """
        pass

    def show_status(self):
        """ Use this function to render status bar at the bottom. """

    @abstractmethod
    def show_gui(self) -> bool:
        """ Define the GUI here. """