import pytest
from rice_app import process_image, load_model
import numpy as np
from PIL import Image

@pytest.fixture
def sample_image():
    return Image.new('RGB', (500, 500), color='red')

def test_process_image(sample_image):
    processed = process_image(sample_image)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1.0

def test_model_loading():
    model = load_model()
    assert hasattr(model, 'predict')
    assert model.input_shape == (None, 224, 224, 3)
