from PIL import Image

from .creation import transforms


class PILImage:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, arr):
        return Image.fromarray(arr, mode=self.mode)


transforms.register_builder("pil_image", PILImage)
