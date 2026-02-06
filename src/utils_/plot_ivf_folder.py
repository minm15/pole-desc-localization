import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_and_plot_broken_axis_clean(folder_path, output_file="ivf_comparison_broken_clean.png"):
    search_pattern = os.path.join(folder_path, "bucket_sizes_*.npy")
    file_list = glob.glob(search_pattern)
    file_list.sort()

    if not file_list:
        print(f"Error: No files found in {folder_path}")
        return

    data_to_plot = []
    labels = []
    all_values = [] 

    print(f"Found {len(file_list)} files.")
    for fpath in file_list:
        try:
            data = np.load(fpath)
            data_to_plot.append(data)
            all_values.extend(data)
            
            filename = os.path.basename(fpath)
            method_name = filename.replace("bucket_sizes_", "").replace(".npy", "")
            display_label = f"{method_name.upper()}"
            labels.append(display_label)
        except Exception as e:
            print(f"Warning: Failed to load {fpath}. {e}")

    if not data_to_plot:
        return

    global_max = np.max(all_values)
    
    threshold_low = 13000 
    
    ylim_top_min = global_max - 2000
    ylim_top_max = global_max + 2000

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [1, 4]})
    
    plt.subplots_adjust(hspace=0.08)  

    flier_settings = dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.6)
    
    median_settings = dict(color="black", linewidth=1.5)

    def draw_boxplot(ax):
        bplot = ax.boxplot(data_to_plot, 
                           vert=True, patch_artist=True, labels=labels,
                           medianprops=median_settings,
                           flierprops=flier_settings)
        
        colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff']
        for patch, color in zip(bplot['boxes'], colors * len(data_to_plot)):
            patch.set_facecolor(color)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        return bplot

    draw_boxplot(ax1) 
    draw_boxplot(ax2) 
    
    ax1.set_ylim(ylim_top_min, ylim_top_max) 
    ax2.set_ylim(0, threshold_low)         

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax1.tick_params(labeltop=False, bottom=False) 
    ax2.xaxis.tick_bottom()

    ax1.set_title("IVF Bucket Distribution Comparison")
    ax2.set_ylabel("Number of Descriptors per Bucket") 

    for i, data in enumerate(data_to_plot):
        q1 = np.percentile(data, 25)
        median = np.median(data)
        q3 = np.percentile(data, 75)
        
        text_str = f"Q3: {q3:.0f}\nMed: {median:.0f}\nQ1: {q1:.0f}"
        
        ax2.text(i + 1.35, median, text_str, 
                 fontsize=9, verticalalignment='center', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    save_path = os.path.join(folder_path, output_file)
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    target_folder = "/home/kaiii/gpu_test/pole-desc-localization/nclt/2013-01-10/profiling/dir_bucket_size" 
    load_and_plot_broken_axis_clean(target_folder)