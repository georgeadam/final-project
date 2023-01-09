from .applicator import Applicator
from .creation import applicators


class Clean(Applicator):
    def apply_corruption(self, x):
        return x


applicators.register_builder("clean", Clean)
