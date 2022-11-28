from PIL import Image

from .creation import transforms


class PILImage:
    def __call__(self, arr):
        return Image.fromarray(arr)


transforms.register_builder("pil_image", PILImage)
