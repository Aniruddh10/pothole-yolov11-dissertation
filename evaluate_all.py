import os
import argparse
import pandas as pd
from ultralytics import YOLO

def evaluate_models(data_yaml, results_dir, output_csv="test_evaluation_results.csv"):
    """
    Evaluates all trained models in the results directory on the test split.
    """
    # Define the models we want to look for and evaluate
    model_configs = {
        "Baseline (50 epochs)": "baseline_yolo11s/weights/best.pt",
        "Baseline (75 epochs)": "baseline_yolo11s_75ep/weights/best.pt",
        "BiFPN (50 epochs)": "bifpn_yolo11s/weights/best.pt",
        "BiFPN (75 epochs)": "bifpn_yolo11s_75ep/weights/best.pt",
        "CBAM (50 epochs)": "cbam_yolo11s/weights/best.pt",
        "CBAM (75 epochs)": "cbam_yolo11s_75ep/weights/best.pt",
        "Combined BiFPN+CBAM (50 epochs)": "bifpn_cbam_yolo11s/weights/best.pt",
        "Combined BiFPN+CBAM (75 epochs)": "bifpn_cbam_yolo11s_75ep/weights/best.pt",
    }

    records = []

    for name, relative_path in model_configs.items():
        weight_path = os.path.join(results_dir, relative_path)
        
        # Check if the weight file actually exists (some runs might not have finished or been renamed)
        if not os.path.exists(weight_path):
            print(f"[-] Checkpoint not found for {name} at {weight_path}. Skipping.")
            continue
            
        print(f"\n[+] Evaluating {name}...")
        try:
            # Load the model directly (the saved .pt contains the full architecture definition)
            model = YOLO(weight_path)
            
            # Run validation on the test split
            # split='test' tells YOLO to run validation on the test set instead of validation set
            metrics = model.val(
                data=data_yaml,
                split="test",
                batch=16,
                imgsz=640,
                plots=True,
                save_json=True,
                verbose=False
            )
            
            # Extract standard metrics
            precision = metrics.results_dict.get("metrics/precision(B)", 0.0)
            recall = metrics.results_dict.get("metrics/recall(B)", 0.0)
            map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)
            
            # Speed metrics (in milliseconds per image)
            speed = metrics.speed
            preprocess_speed = speed.get("preprocess", 0.0)
            inference_speed = speed.get("inference", 0.0)
            postprocess_speed = speed.get("postprocess", 0.0)
            total_speed = preprocess_speed + inference_speed + postprocess_speed

            # Model parameter and GFLOPs count
            params = sum(p.numel() for p in model.model.parameters()) / 1e6 # in millions
            
            records.append({
                "Model Name": name,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "mAP50": round(map50, 4),
                "mAP50-95": round(map50_95, 4),
                "Params (M)": round(params, 2),
                "Pre-process (ms)": round(preprocess_speed, 2),
                "Inference (ms)": round(inference_speed, 2),
                "Post-process (ms)": round(postprocess_speed, 2),
                "Total Latency (ms)": round(total_speed, 2)
            })
            
            print(f"[✓] {name} - mAP50: {map50:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
            
        except Exception as e:
            print(f"[!] Error evaluating {name}: {e}")
            import traceback
            traceback.print_exc()

    if not records:
        print("[-] No models evaluated. Please verify your results directory and checkpoint paths.")
        return

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Saved evaluation results to {output_csv}")
    
    # Print markdown table
    print("\n### Model Performance Comparison (Test Split) ###\n")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 models on test split")
    parser.add_argument("--data", type=str, default="/content/drive/MyDrive/PotholeProject/pothole.yaml", help="Path to pothole.yaml")
    parser.add_argument("--results", type=str, default="/content/drive/MyDrive/PotholeProject/results", help="Directory containing model runs")
    parser.add_argument("--output", type=str, default="test_evaluation_results.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    evaluate_models(args.data, args.results, args.output)
