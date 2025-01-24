import os
import pandas as pd
from src.wikipedia.visualize import visualize
import matplotlib.pyplot as plt
import pytest

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'label': ['good', 'bad', 'good', 'featured', 'featured', 'bad']
    })

def test_visualize_plot(sample_data, tmp_path):
    output_path = tmp_path / "test_plot.png"
    visualize(sample_data, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

