"""
xai.py
Explainability (XAI) ke liye functions — attention heatmaps aur occlusion testing.
Week 3 ke kaam par based hai.
"""
import numpy as np
import torch
from scipy.ndimage import zoom
from PIL import Image


def get_attention_heatmap(image, model, processor, device):
    """
    Vision encoder ki last-layer attention se ek heatmap banata hai,
    jo dikhata hai model ne image ke kis hisse par zyada dhyan diya.
    """
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_attentions=True
        )

    attentions = vision_outputs.attentions[-1]
    attn = attentions[0].mean(dim=0)
    cls_attn = attn[0, 1:]

    num_patches = cls_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    cls_attn_grid = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    heatmap_resized = zoom(
        cls_attn_grid,
        (image.height / grid_size, image.width / grid_size)
    )
    return heatmap_resized


def get_word_attention_heatmaps(image, model, processor, device, max_words=5):
    """
    Caption ke har generated word ke liye alag cross-attention heatmap banata hai.
    Returns: caption (str), words (list), heatmaps (list of arrays)
    """
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, num_beams=1, max_new_tokens=15)
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        decoder_outputs = model.text_decoder(
            input_ids=generated_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
            return_dict=True
        )

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[0])

    last_layer_cross_attn = decoder_outputs.cross_attentions[-1]
    avg_cross_attn = last_layer_cross_attn[0].mean(dim=0)

    grid_size = 24
    heatmaps = []
    words = []
    num_tokens = min(avg_cross_attn.shape[0], max_words)

    for t in range(num_tokens):
        if tokens[t] is None:
            continue
        token_attn = avg_cross_attn[t, 1:]
        attn_grid = token_attn[:grid_size*grid_size].reshape(grid_size, grid_size).cpu().numpy()
        heatmap = zoom(attn_grid, (image.height / grid_size, image.width / grid_size))
        heatmaps.append(heatmap)
        words.append(tokens[t])

    return caption, words, heatmaps


def occlusion_test(image, model, processor, device, grid=4):
    """
    Image ko grid x grid patches mein todta hai, har patch occlude karke
    dekhta hai caption badalta hai ya nahi (LIME-inspired approach).
    """
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, num_beams=1, max_new_tokens=15)
    original_caption = processor.decode(gen_ids[0], skip_special_tokens=True)

    w, h = image.size
    patch_w, patch_h = w // grid, h // grid

    results = []
    for i in range(grid):
        for j in range(grid):
            occluded = image.copy()
            occluded_np = np.array(occluded)
            y1, y2 = i * patch_h, (i + 1) * patch_h
            x1, x2 = j * patch_w, (j + 1) * patch_w
            occluded_np[y1:y2, x1:x2] = 128
            occluded_img = Image.fromarray(occluded_np)

            occ_inputs = processor(occluded_img, return_tensors="pt").to(device)
            with torch.no_grad():
                occ_gen_ids = model.generate(**occ_inputs, num_beams=1, max_new_tokens=15)
            new_caption = processor.decode(occ_gen_ids[0], skip_special_tokens=True)

            changed = new_caption.strip() != original_caption.strip()
            results.append({"patch": (i, j), "caption": new_caption, "changed": changed})

    return original_caption, results
