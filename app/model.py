import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from captum.attr import IntegratedGradients

from .utils import pil_image_to_base64
from .hybrid_mobileNet_ViT import hybrid_model_return, get_class_labels_from_folder  # Your model definition

# Device Setup (Force CPU if needed)
# ─────────────────────────────────────────────────────────────
USE_CPU_ONLY = True  # Set True to avoid CUDA OOM
device = torch.device("cpu" if USE_CPU_ONLY or not torch.cuda.is_available() else "cuda")

# Load Model Once
# ─────────────────────────────────────────────────────────────
model = hybrid_model_return()
model.load_state_dict(torch.load("weights/hybridMobielNet-ViT-Pipeline.pth", map_location=device))
model.to(device)
model.eval()

# Image Transform (Same as Training)
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_LABELS = get_class_labels_from_folder()



# Normalize Tensor for Visualization
# ─────────────────────────────────────────────────────────────
def tensor_to_np(img_tensor):
    img_tensor = img_tensor.squeeze(0).cpu().detach()
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
    return img_tensor.numpy()

# Attribution Map via Captum
# ─────────────────────────────────────────────────────────────
def get_attributions(model, input_img, label_idx):
    input_img = input_img.to(device).unsqueeze(0).requires_grad_()
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_img, target=label_idx, return_convergence_delta=True)
    attributions = attributions.sum(dim=1, keepdim=True)
    return attributions

# Overlay Attribution on Original Image
# ─────────────────────────────────────────────────────────────
def overlay_attribution(orig_img, attribution):
    orig_np = tensor_to_np(orig_img)
    attr_np = attribution.squeeze().detach().cpu().numpy()
    attr_np = np.clip(attr_np, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(orig_np.transpose(1, 2, 0))
    ax.imshow(attr_np, cmap='coolwarm', alpha=0.6)
    ax.axis("off")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Main Inference Function
# ─────────────────────────────────────────────────────────────
def run_inference(base64_img):
    # Step 1: Decode base64 to PIL
    image_data = base64.b64decode(base64_img)
    pil_img = Image.open(BytesIO(image_data)).convert("RGB")

    # Step 2: Transform to tensor
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Step 3: Clear cache (optional)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Step 4: Predict class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        diagnosis_label = CLASS_LABELS.get(predicted_class, f"Class {predicted_class}")
    # Step 5: Attribution map
    attribution = get_attributions(model, input_tensor.squeeze(0), predicted_class)

    # Step 6: Overlay attribution
    processed_img = overlay_attribution(input_tensor.squeeze(0), attribution)

    # Step 7: Convert to base64
    processed_base64 = pil_image_to_base64(processed_img)

    # Step 8: Return result
    diagnosis = f"Diagnosis: {diagnosis_label}"
    return diagnosis, processed_base64
