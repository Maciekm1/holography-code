import numpy as np
import pytest

from Code.functions import bandpass_filter


def test_bandpass_filter_valid_input():
    """Test valid input."""
    # simple grayscale image (2D array)
    image = np.ones((100, 100))  # Uniform image
    small_cutoff = 5
    large_cutoff = 20

    # Call the function
    filtered_image, bandpass_filter_array = bandpass_filter(
        image, small_cutoff, large_cutoff
    )

    # Assert that the outputs are of the correct type and shape
    assert isinstance(
        filtered_image, np.ndarray
    ), "Filtered image should be a numpy array"
    assert isinstance(
        bandpass_filter_array, np.ndarray
    ), "Bandpass filter array should be a numpy array"
    assert (
        filtered_image.shape == image.shape
    ), "Filtered image should have the same shape as input image"
    assert (
        bandpass_filter_array.shape == image.shape
    ), "Bandpass filter array should have the same shape as input image"


def test_bandpass_filter_cutoff_behavior():
    """Test bandpass cutoff behaviour."""
    # simple grayscale image
    image = np.ones((100, 100))  # Uniform image
    small_cutoff = 1  # Very small cutoff
    large_cutoff = 50  # Very large cutoff

    filtered_image, bandpass_filter_array = bandpass_filter(
        image, small_cutoff, large_cutoff
    )

    # Check that the bandpass filter makes sense:
    # Large cutoff size should result in less filtering, any wat to test this?

    assert np.all(
        bandpass_filter_array >= 0
    ), "Bandpass filter array should not have negative values"
    assert (
        np.max(bandpass_filter_array) > 0
    ), "Bandpass filter should allow some frequencies to pass"


def test_bandpass_filter_invalid_input():
    """Test invalid input."""
    # Test for non-2D array input
    image = np.ones((10, 10, 3))  # 3D array
    small_cutoff = 5
    large_cutoff = 20

    with pytest.raises(ValueError, match="Oi Input image should be 2D"):
        bandpass_filter(image, small_cutoff, large_cutoff)

    # Test for invalid cutoff sizes
    image = np.ones((100, 100))
    with pytest.raises(ValueError, match="Cutoff sizes must be positive"):
        bandpass_filter(image, -5, large_cutoff)
    with pytest.raises(ValueError, match="Cutoff sizes must be positive"):
        bandpass_filter(image, small_cutoff, -20)


def test_bandpass_filter_edge_cases():
    """Test edge-cases."""
    # zero image
    image = np.zeros((100, 100))
    small_cutoff = 5
    large_cutoff = 20

    filtered_image, bandpass_filter_array = bandpass_filter(
        image, small_cutoff, large_cutoff
    )

    # Output for a zero image should also be zero
    assert np.all(
        filtered_image == 0
    ), "Filtered image should be zero for a zero input image"
    assert np.all(
        bandpass_filter_array >= 0
    ), "Bandpass filter should not contain negative values"

    # Test with small and large cutoff sizes
    image = np.ones((100, 100))
    small_cutoff = 0.1
    large_cutoff = 200

    filtered_image, bandpass_filter_array = bandpass_filter(
        image, small_cutoff, large_cutoff
    )

    # Verify the output does not throw unexpected errors
    assert (
        filtered_image.shape == image.shape
    ), "Output shape mismatch for extreme cutoff sizes"
    assert (
        bandpass_filter_array.shape == image.shape
    ), "Bandpass filter shape mismatch for extreme cutoff sizes"
