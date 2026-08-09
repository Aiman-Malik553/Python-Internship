"""
model.py
BLIP model load karne aur configure karne ke liye functions.
"""

# Final baseline config jo Week 2 Day 4 mein select hua tha
FINAL_BASELINE_CONFIG = {
    "num_beams": 5,
    "max_new_tokens": 30
}


def get_device():
    """GPU available ho to use karta hai, warna CPU."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_blip_model(model_name="Salesforce/blip-image-captioning-base"):
    """
    BLIP processor aur model dono load karta hai.
    Returns: processor, model, device
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    device = get_device()
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device