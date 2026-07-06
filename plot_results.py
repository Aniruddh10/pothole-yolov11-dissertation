import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparison_plots(csv_path="test_evaluation_results.csv", output_dir="comparison_plots"):
    """
    Generates three beautiful dissertation-ready plots from evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"[-] Evaluation CSV not found: {csv_path}. Please run evaluate_all.py first!")
        return
        
    df = pd.read_csv(csv_path)
    print(f"\n[+] Loaded evaluation data with {len(df)} models. Generating plots...")
    
    # Set styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18
    })

    # --- Plot 1: Bar Chart of mAP50 and mAP50-95 ---
    plt.figure(figsize=(12, 7))
    df_melted = df.melt(id_vars="Model Name", value_vars=["mAP50", "mAP50-95"], 
                        var_name="Metric", value_name="Score")
    
    ax1 = sns.barplot(data=df_melted, x="Score", y="Model Name", hue="Metric", palette="muted")
    ax1.set_title("Performance Comparison (mAP50 vs. mAP50-95)", pad=20, fontweight="bold")
    ax1.set_xlabel("Metric Score")
    ax1.set_ylabel("")
    plt.xlim(0, 1.0)
    plt.legend(title="Metric", loc="lower right")
    
    # Add value annotations on the bars
    for p in ax1.patches:
        width = p.get_width()
        if width > 0:
            ax1.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                     va='center', ha='left', fontsize=10)
                     
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "map_comparison.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved bar chart to: {plot1_path}")

    # --- Plot 2: Precision vs. Recall Scatter Plot (Trade-off) ---
    plt.figure(figsize=(10, 8))
    # We plot the trade-off with a diagonal helper line representing equal F1-score zones
    ax2 = sns.scatterplot(data=df, x="Precision", y="Recall", hue="Model Name", style="Model Name", 
                          s=200, palette="deep")
    
    # Draw curves of constant F1-score (contour levels)
    x_range = np.linspace(0.4, 0.95, 100)
    y_range = np.linspace(0.4, 0.95, 100)
    X, Y = np.meshgrid(x_range, y_range)
    F1 = 2 * (X * Y) / (X + Y)
    contours = plt.contour(X, Y, F1, levels=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75], colors="gray", alpha=0.3, linestyles="dashed")
    plt.clabel(contours, inline=True, fontsize=10, fmt='F1=%.2f')

    ax2.set_title("Precision vs. Recall Trade-off Map", pad=20, fontweight="bold")
    ax2.set_xlabel("Precision")
    ax2.set_ylabel("Recall")
    plt.xlim(0.4, 0.95)
    plt.ylim(0.4, 0.95)
    
    # Label each point with a small offset
    for idx, row in df.iterrows():
        plt.text(row["Precision"] + 0.005, row["Recall"] + 0.005, row["Model Name"], fontsize=9, alpha=0.8)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model Variant")
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "precision_recall_scatter.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved scatter plot to: {plot2_path}")

    # --- Plot 3: Accuracy vs. Inference Latency (Efficiency Frontier) ---
    plt.figure(figsize=(11, 8))
    # Marker size represents parameters in millions
    ax3 = sns.scatterplot(data=df, x="Total Latency (ms)", y="mAP50", hue="Model Name", style="Model Name",
                          size="Params (M)", sizes=(100, 500), palette="Set1")
    
    ax3.set_title("Latency vs. Accuracy (Efficiency Map)", pad=20, fontweight="bold")
    ax3.set_xlabel("Total Latency (ms/image)")
    ax3.set_ylabel("mAP50 Score")
    
    # Annotate points
    for idx, row in df.iterrows():
        plt.text(row["Total Latency (ms)"] + 0.1, row["mAP50"] + 0.005, 
                 f'{row["Model Name"]}\n({row["Params (M)"]}M params)', fontsize=8, alpha=0.8)
                 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model & Size")
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "latency_vs_accuracy.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved efficiency map to: {plot3_path}")

if __name__ == "__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description="Generate beautiful comparative plots from evaluation results")
    parser.add_argument("--csv", type=str, default="test_evaluation_results.csv", help="Path to evaluation results CSV")
    parser.add_argument("--output", type=str, default="comparison_plots", help="Directory to save plots")
    
    args = parser.parse_args()
    generate_comparison_plots(args.csv, args.output)
