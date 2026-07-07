import os
import cv2
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

def compare_predictions(image_path, results_dir, output_dir="comparison_results"):
    """
    Runs inference on an image using four main models and saves a side-by-side comparison grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 75-epoch variants are chosen as they represent the fully converged models
    model_paths = {
        "Baseline": "baseline_yolo11s_75ep/weights/best.pt",
        "BiFPN (Neck)": "bifpn_yolo11s_75ep/weights/best.pt",
        "CBAM (Attention)": "cbam_yolo11s_75ep/weights/best.pt",
        "Combined (BiFPN+CBAM)": "bifpn_cbam_yolo11s_75ep/weights/best.pt"
    }
    
    # Read original image
    if not os.path.exists(image_path):
        print(f"[-] Image not found: {image_path}")
        return
        
    img_name = os.path.basename(image_path)
    print(f"\n[+] Comparing model predictions on: {img_name}")
    
    # Load original image for plotting
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Setup matplotlib figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, (name, rel_path) in enumerate(model_paths.items()):
        weight_path = os.path.join(results_dir, rel_path)
        
        if not os.path.exists(weight_path):
            # Try case-insensitive folder search
            parts = rel_path.split("/")
            if len(parts) >= 2 and os.path.exists(results_dir):
                folder_name = parts[0]
                sub_path = "/".join(parts[1:])
                found = False
                for entry in os.listdir(results_dir):
                    if entry.lower() == folder_name.lower():
                        alternative_path = os.path.join(results_dir, entry, sub_path)
                        if os.path.exists(alternative_path):
                            weight_path = alternative_path
                            found = True
                            break
                if not found:
                    print(f"[-] Checkpoint not found for {name} (tried case-insensitive search). Using placeholder.")
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"{name} (No Checkpoint)", fontsize=14, color="red")
                    axes[i].axis("off")
                    continue
            else:
                print(f"[-] Checkpoint not found for {name} at {weight_path}. Using placeholder.")
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"{name} (No Checkpoint)", fontsize=14, color="red")
                axes[i].axis("off")
                continue
            
        try:
            # Load and run inference
            model = YOLO(weight_path)
            results = model.predict(image_path, imgsz=640, verbose=False)
            
            # Plot the prediction result (YOLO plot() returns a BGR numpy array)
            pred_plot = results[0].plot()
            pred_plot_rgb = cv2.cvtColor(pred_plot, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(pred_plot_rgb)
            # Count detected objects
            det_count = len(results[0].boxes)
            axes[i].set_title(f"{name} (Detections: {det_count})", fontsize=14, fontweight="bold")
            axes[i].axis("off")
            print(f"[✓] Run inference with {name}: {det_count} detections.")
            
        except Exception as e:
            print(f"[!] Error running inference with {name}: {e}")
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{name} (Inference Error)", fontsize=14, color="red")
            axes[i].axis("off")

    plt.tight_layout()
    out_img_path = os.path.join(output_dir, f"compare_{img_name}")
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved comparison figure to: {out_img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLO11 model predictions side-by-side")
    parser.add_argument("--image", type=str, required=True, help="Path to input test image")
    parser.add_argument("--results", type=str, default="/content/drive/MyDrive/PotholeProject/results", help="Directory containing model runs")
    parser.add_argument("--output", type=str, default="comparison_results", help="Directory to save output comparison grid")
    
    args = parser.parse_args()
    compare_predictions(args.image, args.results, args.output)
