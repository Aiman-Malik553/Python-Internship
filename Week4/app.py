import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()


def get_attention_heatmap(image):
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
    heatmap = zoom(cls_attn_grid, (image.height / grid_size, image.width / grid_size))
    return heatmap


def generate_caption_and_heatmap(image):
    if image is None:
        return "Please upload an image.", None

    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=5, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    heatmap = get_attention_heatmap(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_image = Image.open(buf)

    return caption, overlay_image


demo = gr.Interface(
    fn=generate_caption_and_heatmap,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Textbox(label="Generated Caption (BLIP)"),
        gr.Image(label="Grad-CAM Attention Overlay")
    ],
    title="Image Captioning with Explainability (BLIP)",
    description="Upload an image to get a caption from BLIP, along with a heatmap showing which regions the model attended to."
)

if __name__ == "__main__":
    demo.launch(share=True)
