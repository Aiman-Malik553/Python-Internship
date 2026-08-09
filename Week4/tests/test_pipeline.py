"""
tests/test_pipeline.py
Basic unit tests for data_loader.py and inference.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from PIL import Image


def test_load_captions_returns_dataframe(tmp_path):
    from data_loader import load_captions
    csv_file = tmp_path / "captions.csv"
    csv_file.write_text("image,caption\ntest.jpg,a test caption\n")
    df = load_captions(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert "image" in df.columns
    assert "caption" in df.columns


def test_get_human_captions_returns_list():
    from data_loader import get_human_captions
    df = pd.DataFrame({
        "image": ["a.jpg", "a.jpg", "b.jpg"],
        "caption": ["cap1", "cap2", "cap3"]
    })
    result = get_human_captions(df, "a.jpg")
    assert isinstance(result, list)
    assert len(result) == 2


def test_get_human_captions_empty_for_missing_image():
    from data_loader import get_human_captions
    df = pd.DataFrame({"image": ["a.jpg"], "caption": ["cap1"]})
    result = get_human_captions(df, "nonexistent.jpg")
    assert result == []


def test_list_all_images_returns_list(tmp_path):
    from data_loader import list_all_images
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.jpg").touch()
    result = list_all_images(str(tmp_path))
    assert isinstance(result, list)
    assert len(result) == 2


def test_final_baseline_config_has_required_keys():
    from model import FINAL_BASELINE_CONFIG
    assert "num_beams" in FINAL_BASELINE_CONFIG
    assert "max_new_tokens" in FINAL_BASELINE_CONFIG
    assert FINAL_BASELINE_CONFIG["num_beams"] == 5
