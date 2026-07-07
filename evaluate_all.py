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
            # Try case-insensitive folder search
            parts = relative_path.split("/")
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
                    print(f"[-] Checkpoint not found for {name} (tried case-insensitive search). Skipping.")
                    continue
            else:
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
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            
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
                "F1-Score": round(f1_score, 4),
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
    
    # Generate Ablation Study Tables
    ablation_mapping = {
        "Baseline (50 epochs)": {"Configuration": "Baseline", "BiFPN": "✗", "CBAM": "✗", "Epochs": 50},
        "BiFPN (50 epochs)": {"Configuration": "+ BiFPN", "BiFPN": "✓", "CBAM": "✗", "Epochs": 50},
        "CBAM (50 epochs)": {"Configuration": "+ CBAM", "BiFPN": "✗", "CBAM": "✓", "Epochs": 50},
        "Combined BiFPN+CBAM (50 epochs)": {"Configuration": "Proposed (BiFPN+CBAM)", "BiFPN": "✓", "CBAM": "✓", "Epochs": 50},
        "Baseline (75 epochs)": {"Configuration": "Baseline", "BiFPN": "✗", "CBAM": "✗", "Epochs": 75},
        "BiFPN (75 epochs)": {"Configuration": "+ BiFPN", "BiFPN": "✓", "CBAM": "✗", "Epochs": 75},
        "CBAM (75 epochs)": {"Configuration": "+ CBAM", "BiFPN": "✗", "CBAM": "✓", "Epochs": 75},
        "Combined BiFPN+CBAM (75 epochs)": {"Configuration": "Proposed (BiFPN+CBAM)", "BiFPN": "✓", "CBAM": "✓", "Epochs": 75},
    }
    
    ablation_rows = []
    for _, row in df.iterrows():
        model_name = row["Model Name"]
        if model_name in ablation_mapping:
            info = ablation_mapping[model_name]
            new_row = {
                "Configuration": info["Configuration"],
                "BiFPN": info["BiFPN"],
                "CBAM": info["CBAM"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1-Score": row["F1-Score"],
                "mAP50": row["mAP50"],
                "mAP50-95": row["mAP50-95"],
                "Params (M)": row["Params (M)"],
                "Latency (ms)": row["Total Latency (ms)"],
                "Epochs": info["Epochs"]
            }
            ablation_rows.append(new_row)
            
    if ablation_rows:
        df_ab = pd.DataFrame(ablation_rows)
        
        md_content = "# Ablation Study Results\n\n"
        latex_content = "% --- Ablation Study LaTeX Tables ---\n\n"
        
        for epochs in [50, 75]:
            df_epoch = df_ab[df_ab["Epochs"] == epochs].copy()
            if df_epoch.empty:
                continue
                
            # Drop the Epochs column for presentation
            df_epoch_clean = df_epoch.drop(columns=["Epochs"])
            
            # 1. Append to Markdown file
            md_content += f"## Ablation Study ({epochs} Epochs)\n\n"
            md_content += df_epoch_clean.to_markdown(index=False) + "\n\n"
            
            # 2. Append to LaTeX file
            latex_table = []
            latex_table.append(r"\begin{table}[h]")
            latex_table.append(r"  \centering")
            latex_table.append(r"  \caption{Ablation Study of Proposed YOLO11 Modifications (" + str(epochs) + " Epochs)}")
            latex_table.append(r"  \label{tab:ablation_" + str(epochs) + "}")
            latex_table.append(r"  \begin{tabular}{l|cc|ccccc|c|r}")
            latex_table.append(r"    \hline")
            latex_table.append(r"    Configuration & BiFPN & CBAM & Precision & Recall & F1-Score & mAP50 & mAP50-95 & Params (M) & Latency (ms) \\")
            latex_table.append(r"    \hline")
            
            for _, r in df_epoch_clean.iterrows():
                bifpn_sym = r"\checkmark" if r["BiFPN"] == "✓" else r"\times"
                cbam_sym = r"\checkmark" if r["CBAM"] == "✓" else r"\times"
                latex_table.append(
                    f"    {r['Configuration']} & {bifpn_sym} & {cbam_sym} & "
                    f"{r['Precision']:.4f} & {r['Recall']:.4f} & {r['F1-Score']:.4f} & "
                    f"{r['mAP50']:.4f} & {r['mAP50-95']:.4f} & "
                    f"{r['Params (M)']:.2f} & {r['Latency (ms)']:.2f} \\\\"
                )
            
            latex_table.append(r"    \hline")
            latex_table.append(r"  \end{tabular}")
            latex_table.append(r"\end{table}")
            
            latex_content += "\n".join(latex_table) + "\n\n"
            
            # Print to console
            print(f"\n### Ablation Study ({epochs} Epochs) ###\n")
            print(df_epoch_clean.to_markdown(index=False))
            
        # Write files
        with open("ablation_study.md", "w") as f:
            f.write(md_content)
        with open("ablation_study.tex", "w") as f:
            f.write(latex_content)
            
        print("\n[✓] Generated 'ablation_study.md' and 'ablation_study.tex' (for copy-pasting directly into Overleaf/LaTeX)!")
    else:
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
