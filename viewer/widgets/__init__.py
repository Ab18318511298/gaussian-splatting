import torch
from viewer.types import ViewerMode
from abc import ABC, abstractmethod

class Widget(ABC):
    id = 0

    def __init__(self, mode: ViewerMode):
        self.mode = mode
        self.widget_id = self.id
        self.id += 1

    def setup(self):
        """
        Perform any setup actions required after OpenGL/GLFW/ImGUI is initialized. 
        This function won't be called when application is running in headless mode.
        """

    def destroy(self):
        """
        Destroy any resources created manually in the 'setup' function.
        This function won't be called when application is running in headless mode.
        """

    def server_send(self):
        """
        Send widget state to the client.

        Returns:
            binary (bytes): Any binary data to be sent to the client.
            text (dict): Any text data to be sent to the client.
        """
        return None, None
    
    def server_recv(self, binary, text):
        """
        Receive widget state from the client and update it.

        Args:
            binary (bytes): Any binary data received from the client.
            text (dict): Any text data sent received from the client
        """
    
    def client_send(self):
        """
        Send widget state to the server.

        Returns:
            binary (bytes): Any binary data to be sent to the server.
            text (dict): Any text data to be sent to the server.
        """
        return None, None

    def client_recv(self, binary, dict):
        """
        Send widget state to the server.

        Args:
            binary (bytes): Any binary data to be received from the server.
            text (dict): Any text data to be received from the server.
        """
    
    @abstractmethod
    def show_gui(self):
        pass