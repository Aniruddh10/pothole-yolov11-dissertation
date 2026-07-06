import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_curves(results_dir, output_dir="training_plots"):
    """
    Reads the results.csv files from each model's training directory and plots
    comparative training curves (loss and mAP50) over epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define directories containing training logs
    model_dirs = {
        "Baseline (50ep)": "baseline_yolo11s",
        "Baseline (75ep)": "baseline_yolo11s_75ep",
        "BiFPN (50ep)": "bifpn_yolo11s",
        "BiFPN (75ep)": "bifpn_yolo11s_75ep",
        "CBAM (50ep)": "cbam_yolo11s",
        "CBAM (75ep)": "cbam_yolo11s_75ep",
        "Combined (50ep)": "bifpn_cbam_yolo11s",
        "Combined (75ep)": "bifpn_cbam_yolo11s_75ep"
    }

    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    # Prepare figures: 1 for mAP50, 1 for box loss, 1 for class loss
    fig_map, ax_map = plt.subplots(figsize=(10, 6))
    fig_box, ax_box = plt.subplots(figsize=(10, 6))
    fig_cls, ax_cls = plt.subplots(figsize=(10, 6))
    
    found_any = False

    for label, folder in model_dirs.items():
        csv_path = os.path.join(results_dir, folder, "results.csv")
        
        if not os.path.exists(csv_path):
            print(f"[-] Training log results.csv not found at: {csv_path}. Skipping.")
            continue
            
        print(f"[+] Loading training curves for {label}...")
        try:
            df = pd.read_csv(csv_path)
            # Strip whitespace from column headers
            df.columns = df.columns.str.strip()
            
            # Verify columns exist
            epoch_col = "epoch"
            map_col = "metrics/mAP50(B)"
            box_loss_col = "train/box_loss"
            cls_loss_col = "train/cls_loss"
            
            if epoch_col not in df.columns:
                print(f"[!] Warning: 'epoch' column not found in {csv_path}. Skipping.")
                continue
                
            found_any = True
            
            # 1. Plot mAP50 over epochs
            if map_col in df.columns:
                ax_map.plot(df[epoch_col], df[map_col], label=label, linewidth=2)
                
            # 2. Plot Training Box Loss
            if box_loss_col in df.columns:
                ax_box.plot(df[epoch_col], df[box_loss_col], label=label, linewidth=2)
                
            # 3. Plot Training Class Loss
            if cls_loss_col in df.columns:
                ax_cls.plot(df[epoch_col], df[cls_loss_col], label=label, linewidth=2)
                
        except Exception as e:
            print(f"[!] Error loading {csv_path}: {e}")

    if not found_any:
        print("[-] No training logs were loaded. Make sure your results directory has the results.csv files.")
        plt.close(fig_map)
        plt.close(fig_box)
        plt.close(fig_cls)
        return

    # Finish and save validation mAP50 plot
    ax_map.set_title("Validation mAP50 Comparison over Epochs", pad=15, fontweight="bold")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP50 Score")
    ax_map.legend(loc="lower right")
    fig_map.tight_layout()
    map_path = os.path.join(output_dir, "train_curve_map50.png")
    fig_map.savefig(map_path, dpi=300, bbox_inches="tight")
    plt.close(fig_map)
    print(f"[✓] Saved validation mAP50 curve to: {map_path}")

    # Finish and save training box loss plot
    ax_box.set_title("Training Box Loss Progression", pad=15, fontweight="bold")
    ax_box.set_xlabel("Epoch")
    ax_box.set_ylabel("Loss")
    ax_box.legend(loc="upper right")
    fig_box.tight_layout()
    box_path = os.path.join(output_dir, "train_curve_box_loss.png")
    fig_box.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close(fig_box)
    print(f"[✓] Saved training box loss curve to: {box_path}")

    # Finish and save training class loss plot
    ax_cls.set_title("Training Class Loss Progression", pad=15, fontweight="bold")
    ax_cls.set_xlabel("Epoch")
    ax_cls.set_ylabel("Loss")
    ax_cls.legend(loc="upper right")
    fig_cls.tight_layout()
    cls_path = os.path.join(output_dir, "train_curve_cls_loss.png")
    fig_cls.savefig(cls_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cls)
    print(f"[✓] Saved training class loss curve to: {cls_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot comparative training curves from YOLO results.csv logs")
    parser.add_argument("--results", type=str, default="/content/drive/MyDrive/PotholeProject/results", help="Directory containing model runs")
    parser.add_argument("--output", type=str, default="training_plots", help="Directory to save plots")
    
    args = parser.parse_args()
    plot_training_curves(args.results, args.output)
