import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Global dictionary to store captured attention maps
spatial_attention_maps = []
layer_names = []

def register_hooks(model):
    """
    Registers forward hooks on all SpatialAttention modules' Sigmoid activation layers.
    """
    global spatial_attention_maps, layer_names
    spatial_attention_maps.clear()
    layer_names.clear()

    def get_activation_hook(layer_name):
        def hook(module, input, output):
            # output is the sigmoid activation of shape (B, 1, H, W)
            spatial_attention_maps.append((layer_name, output.detach().cpu()))
        return hook

    # Traverse the model to find CBAM modules and their SpatialAttention sub-modules
    # Standard YOLO model structure is model.model.model (ModuleList)
    for idx, layer in enumerate(model.model.model):
        class_name = layer.__class__.__name__
        if class_name == "CBAM":
            print(f"[+] Registering hook for CBAM at Layer {idx}")
            # Register hook on the Sigmoid activation of SpatialAttention
            layer.spatial_attention.act.register_forward_hook(get_activation_hook(f"Layer {idx} (CBAM)"))

def visualize_attention(image_path, weight_path, output_dir="attention_results"):
    """
    Runs inference on an image and saves the overlaid CBAM spatial attention heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return
        
    if not os.path.exists(weight_path):
        print(f"[-] Model checkpoint not found at: {weight_path}")
        return

    img_name = os.path.basename(image_path)
    print(f"\n[+] Visualizing CBAM attention for image: {img_name}")
    
    # Load model and register hooks
    model = YOLO(weight_path)
    register_hooks(model)
    
    # Read original image
    orig_img = cv2.imread(image_path)
    h, w, c = orig_img.shape
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Run forward pass (inference)
    # This will trigger our hooks and populate spatial_attention_maps
    results = model.predict(image_path, imgsz=640, verbose=False)
    
    if not spatial_attention_maps:
        print("[-] No attention maps were captured. Is this a CBAM model?")
        return
        
    print(f"[✓] Captured {len(spatial_attention_maps)} spatial attention maps.")
    
    # For each captured attention map, save a visualization
    for name, attn_tensor in spatial_attention_maps:
        # attn_tensor shape is (1, 1, H_feat, W_feat)
        attn_map = attn_tensor[0, 0].numpy()
        
        # Normalize to 0-255 range
        attn_map_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map_255 = (attn_map_normalized * 255).astype(np.uint8)
        
        # Resize to original image dimensions
        attn_map_resized = cv2.resize(attn_map_255, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap to get a colored heatmap (Jet color scheme)
        heatmap = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        alpha = 0.55 # transparency factor
        overlay = cv2.addWeighted(orig_img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
        
        # Plot and save
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(orig_img_rgb)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")
        
        # Plot raw heatmap
        axes[1].imshow(heatmap_rgb)
        axes[1].set_title(f"Spatial Attention Heatmap ({name})", fontsize=14)
        axes[1].axis("off")
        
        # Plot overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Attention Overlay", fontsize=14)
        axes[2].axis("off")
        
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        out_filename = f"attention_{safe_name}_{img_name}"
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved attention visualization to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CBAM spatial attention heatmaps")
    parser.add_argument("--image", type=str, required=True, help="Path to input test image")
    parser.add_argument("--model", type=str, required=True, help="Path to CBAM or Combined model weight file")
    parser.add_argument("--output", type=str, default="attention_results", help="Directory to save visual results")
    
    args = parser.parse_args()
    visualize_attention(args.image, args.model, args.output)
