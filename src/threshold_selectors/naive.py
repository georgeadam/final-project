import numpy as np

from src.utils.metric import get_metrics
from .creation import threshold_selectors
from .threshold_selector import ThresholdSelector


class Naive(ThresholdSelector):
    def _select_threshold_helper(self, probs, y):
        if self.desired_rate != "f1":
            return self._binary_search(probs, y)
        else:
            return self._linear_search(probs, y)

    def _binary_search(self, probs, y):
        thresholds = np.linspace(0.01, 0.999, 1000)
        best_threshold = None
        best_diff = float("inf")

        l = 0
        r = len(thresholds)

        while l < r:
            mid = l + (r - l) // 2
            threshold = thresholds[mid]

            temp_preds = probs[:, 1] >= threshold
            rates = get_metrics(probs, temp_preds, y)

            direction = self._get_direction()

            if abs(rates[self.desired_rate] - self.desired_value) <= self.desired_rate * self.tolerance:
                return threshold
            elif (rates[self.desired_rate] < self.desired_value and direction == "decreasing") or (
                    rates[self.desired_rate] > self.desired_value and direction == "increasing"):
                l = mid + 1
            else:
                r = mid - 1

            if abs(rates[self.desired_rate] - self.desired_value) <= best_diff:
                best_diff = abs(rates[self.desired_rate] - self.desired_value)
                best_threshold = threshold

        return best_threshold

    def _linear_search(self, probs, y):
        thresholds = np.linspace(0.01, 0.999, 1000)
        best_threshold = None
        best_diff = float("inf")

        for threshold in thresholds:
            temp_preds = probs[:, 1] >= threshold

            rates = get_metrics(probs, temp_preds, y)

            if abs(rates[self.desired_rate] - self.desired_value) <= best_diff:
                best_diff = abs(rates[self.desired_rate] - self.desired_value)
                best_threshold = threshold

        return best_threshold

    def _get_direction(self) -> str:
        if self.desired_rate == "fpr" or self.desired_rate == "tnr":
            return "increasing"
        elif self.desired_rate == "fnr" or self.desired_rate == "tpr":
            return "decreasing"


threshold_selectors.register_builder("naive", Naive)
