"""
inference.py
BLIP model se caption generate karne ke liye functions.
"""
from model import FINAL_BASELINE_CONFIG


def generate_caption(image, processor, model, device, config=None):
    """
    Ek image ke liye caption generate karta hai.
    config na diya jaye to Week 2 wala final baseline config use hota hai.
    """
    if config is None:
        config = FINAL_BASELINE_CONFIG

    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, **config)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_captions_batch(images_dict, processor, model, device, config=None):
    """
    Multiple images ke liye caption generate karta hai.
    images_dict: {"image_name": PIL_Image, ...}
    Returns: {"image_name": "caption", ...}
    """
    results = {}
    for name, image in images_dict.items():
        results[name] = generate_caption(image, processor, model, device, config)
    return results
