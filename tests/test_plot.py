# Tech preamble:
import numpy as np
import pytest

# Import functions to test:
from predict_sahel_rainfall.plot import bar_color


# Define test data:
test_data = np.array([1.0, -0.5, 3.0, 0.0])


def test_bar_color():
    """Test, if function returns correct color codes according to values' signs."""
    assert all(
        bar_color(data=test_data, color_pos="b", color_neg="r")
        == np.array(["b", "r", "b", "r"])
    )
