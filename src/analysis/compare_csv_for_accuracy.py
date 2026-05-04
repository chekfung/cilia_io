import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import statsmodels.api as sm

power_analysis = TTestIndPower()

DORSAL_COLOR = (179, 110, 59)
VENTRAL_COLOR = (132, 172, 195)

def cv2_to_rgb(bgr):
    return tuple([c/255.0 for c in bgr[::-1]])

PALETTE = [cv2_to_rgb(DORSAL_COLOR), cv2_to_rgb(VENTRAL_COLOR)]

MASTER_MAP = {
    "Dorsal": PALETTE[0], 
    "Ventral": PALETTE[1]
}

MASTER_MAP_VIOLIN = {
    0: PALETTE[0], 
    1: PALETTE[1]
}

def significance_stars(p_val, alpha=0.05):
    if p_val < 0.0001:
        return "****"
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < alpha:
        return "*"
    else:
        return "ns"
    
def calculate_nested_power(df, feature, alpha=0.05, target_power=0.8):
    """
    Calculates power based on fish rather than cilia.
    """
    # collapse data to fish medians 
    fish_medians = df.groupby(['fname', 'group'])[feature].median().reset_index()
    
    g0_medians = fish_medians[fish_medians['group'] == 0][feature]
    g1_medians = fish_medians[fish_medians['group'] == 1][feature]
    
    n0, n1 = len(g0_medians), len(g1_medians)
    
    if n0 < 2 or n1 < 2:
        return "Insufficient Fish N for Power Analysis"

    # Calculate Cohen's d based on Fish-to-Fish variation
    mean_diff = np.mean(g0_medians) - np.mean(g1_medians)
    pooled_std = np.sqrt(((n0-1)*np.var(g0_medians, ddof=1) + (n1-1)*np.var(g1_medians, ddof=1)) / (n0+n1-2))
    
    if pooled_std == 0: return 0, 0
    
    effect_size = abs(mean_diff) / pooled_std
    
    # 3. Solve for required Biological N
    analysis = TTestIndPower()
    required_fish_n = analysis.solve_power(effect_size=effect_size, 
                                           alpha=alpha, 
                                           power=target_power, 
                                           ratio=n1/n0)
    return effect_size, required_fish_n

def plot_violin_and_utest(df, feature, name, fname="", feat_unit=None):
    g0 = df[df['group'] == 0][feature].dropna()
    g1 = df[df['group'] == 1][feature].dropna()

    # Mann-Whitney U for cilia-level
    u_stat, u_test_pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    u_stars = significance_stars(u_test_pval)

    # Paired Wilcoxon for fish-level medians, std, cv, mean
    fish_stats = df.groupby(['fname', 'group'])[feature].agg(['median', 'std', 'mean']).reset_index()
    fish_stats['cv'] = fish_stats['std'] / fish_stats['mean']
    
    paired_df = fish_stats.pivot(index='fname', columns='group', values='median').dropna()
    
    if paired_df.empty:
        print(f"Warning: No paired samples found for {feature} in {fname}")
        return

    g0_meds, g1_meds = paired_df[0], paired_df[1]
    stat, p_val = stats.wilcoxon(g0_meds, g1_meds)
    stars = significance_stars(p_val)

    spread_df = fish_stats.pivot(index='fname', columns='group', values='std').dropna()
    if not spread_df.empty:
        _, p_var_spread = stats.wilcoxon(spread_df[0], spread_df[1])
        stars_var = significance_stars(p_var_spread)
    else:
        p_var_spread, stars_var = np.nan, "ns"

    cv_df = fish_stats.pivot(index='fname', columns='group', values='cv').dropna()
    if not cv_df.empty:
        _, p_cv = stats.wilcoxon(cv_df[0], cv_df[1])
        stars_cv = significance_stars(p_cv)
    else:
        p_cv = np.nan
        stars_cv = "ns"

    n1, n2 = len(g0), len(g1)
    mean_diff = np.mean(g1) - np.mean(g0)
    s1, s2 = np.std(g0, ddof=1), np.std(g1, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    cohen_d_cilia = mean_diff / (pooled_std if pooled_std != 0 else 1)
    required_n_cilia = power_analysis.solve_power(effect_size=max(abs(cohen_d_cilia), 0.0001),
                                                 alpha=0.05, power=0.8, ratio=n2/n1)

    meow = calculate_nested_power(df, feature)
    effect_size_fish = 0
    required_n_fish = 0

    point_size = 10
    iteration, max_iterations = 0, 50
    plot_fish_meds = fish_stats.rename(columns={'median': feature})

    while iteration < max_iterations:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plt.figure(figsize=(3.5, 4.5))
            ax = plt.gca()
            
            sns.violinplot(x="group", y=feature, data=df, hue="group",
                           palette=MASTER_MAP_VIOLIN, inner=None, cut=0, alpha=0.3, legend=False)

            sns.swarmplot(x="group", y=feature, data=df, hue="fname",
                          palette="tab10", dodge=False, size=point_size, ax=ax, alpha=0.3, zorder=2)

            # Fish Medians (Diamonds)
            sns.stripplot(x="group", y=feature, data=plot_fish_meds, hue="fname",
                          palette="tab10", marker="D", size=7, jitter=0.1, 
                          edgecolor="black", linewidth=0.8, ax=ax, zorder=15, alpha=0.8)

            # Iteratively reduce size of points until we can place all of them
            if not any("points cannot be placed" in str(warning.message) for warning in w):
                sns.boxplot(x="group", y=feature, data=df, ax=ax, width=0.1, showfliers=False,
                            boxprops={'facecolor': '#333333', 'zorder': 10, 'edgecolor': 'none', 'alpha':0.8},
                            whiskerprops={'color': '#333333', 'alpha':0.8},
                            medianprops={'color': 'white', 'linewidth': 3, 'zorder':20},
                            capprops={'linewidth': 0})
                break
            else:
                plt.close(); point_size *= 0.8; iteration += 1

    if ax.get_legend() is not None: ax.get_legend().remove()

    plt.xticks([0, 1], ["Dorsal", "Ventral"])
    unit_str = f" ({feat_unit})" if feat_unit else ""
    plt.title(f"{feature.replace('_',' ').title()} {name}", fontsize=11)
    plt.ylabel(feature.replace("_", " ").title() + unit_str)
    plt.xlabel("")
    sns.despine()

    y_max, y_min = df[feature].max(), df[feature].min()
    y, h = y_max + 0.05*(y_max-y_min), 0.03*(y_max-y_min)
    if stars != "ns":
        ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.2, c="k")
        ax.text(0.5, y+h, stars, ha="center", va="bottom", fontsize=12, fontweight='bold')

    print(f"--- {feature} ({name}) ---")
    print(f"Location Stats : Wilcoxon (Fish Medians) p={p_val:.4e} ({stars})")
    print(f"Variance Stats : Paired SD p={p_var_spread:.4e} ({stars_var}) | Paired CV p={p_cv:.4e} ({stars_cv})")
    print(f"Cilia Stats    : U-Test p={u_test_pval:.4e} ({u_stars})")
    print(f"Power (Cilia)  : Cohen'd={cohen_d_cilia:.2f} (Req.N={required_n_cilia:.1f})")
    print(f"Power (Fish)   : Cohen'd={effect_size_fish:.2f} (Req.N={required_n_fish:.1f})")

    SAVE_FIGURES = True

    if SAVE_FIGURES:
        plt.savefig(os.path.join("test", f"{name}_{feature}_paired.png"), 
                    dpi=300, bbox_inches='tight')

def calculate_metrics(df, label="Group", include_length=True, include_frequency=True):
    """Calculates metrics for frequency and optionally length."""
    if include_frequency:
        f1, f2 = df["frequency_1"], df["frequency_2"]
        f_mse = mean_squared_error(f1, f2)
        f_mape = mean_absolute_percentage_error(f1, f2) * 100
        
        # Calculate STD of errors
        f_abs_errors = (f1 - f2).abs()
        f_ape = (f_abs_errors / f1) * 100
        f_std_abs = f_abs_errors.std()
        f_std_ape = f_ape.std()
        
        print(f"--- {label} ---")
        print(f"   Samples: {len(df)}")
        print(f"   Freq MSE: {f_mse:.4f} | Freq MAPE: {f_mape:.2f}%")
        print(f"   Freq Error STD: {f_std_abs:.4f} Hz | Freq APE STD: {f_std_ape:.2f}%")
    
    if include_length:
        l1, l2 = df["mean_cilia_length_1"], df["mean_cilia_length_2"]
        l_mse = mean_squared_error(l1, l2)
        l_mape = mean_absolute_percentage_error(l1, l2) * 100

        # Calculate STD of errors
        l_abs_errors = (l1 - l2).abs()
        l_ape = (l_abs_errors / l1) * 100
        l_std_abs = l_abs_errors.std()
        l_std_ape = l_ape.std()

        if not include_frequency: print(f"--- {label} ---")
        print(f"   Samples: {len(df)}")
        print(f"   Len MSE: {l_mse:.4f} | Len MAPE: {l_mape:.2f}%")
        print(f"   Len Error STD: {l_std_abs:.4f} um | Len APE STD: {l_std_ape:.2f}%")
    print("-" * 50)

# --- pairs to compare ---
# Note: for thouvenin, all use the size of 4, ROI=40.
thouvenin_pairs = [
    ("../../data/4_13_26_thouvenin_comparison_data/bbs2_21_f_vent_thouvenin_label.csv", 
     "../../data/10_29_2025_4dpf_annotated/bbs2_21_f_vent_quant_skel_center.csv"),
    ("../../data/4_13_26_thouvenin_comparison_data/wt_17_f_vent001_thouvenin_label.csv", 
     "../../data/10_29_2025_4dpf_annotated/wt_17_f_vent001_quant_skel_center.csv"),
   ("../../data/4_13_26_thouvenin_comparison_data/wt_24_f_vent002_thouvenin_label.csv", 
     "../../data/10_29_2025_4dpf_annotated/wt_24_f_vent002_quant_skel_center.csv")

]

manual_pairs = [
    ('../../data/10_30_2025_comparison_data/wt_22_f_vent001__manual_label_ece.csv', 
     "../../data/10_30_2025_comparison_data/wt_22_f_vent001_quant_skel_center.csv"),
     ('../../data/10_30_2025_comparison_data/bbs2_24_f_vent001_manual_label_ece.csv', 
     "../../data/10_29_2025_4dpf_annotated/bbs2_24_f_vent001_quant_skel_center.csv"),
     ('../../data/10_30_2025_comparison_data/wt_15_f_vent001_manual_label_ece.csv', 
     "../../data/10_29_2025_4dpf_annotated/wt_15_f_vent001_quant_skel_center.csv")
]

print("PROCESSING THOUVENIN VIDEOS\n")
thou_dfs = []
for t_path, q_path in thouvenin_pairs:
    vid_name = os.path.basename(t_path).replace("_thouvenin_label.csv", "")
    d1 = pd.read_csv(t_path, index_col="cilia_id")  # thouvenin 
    d2 = pd.read_csv(q_path, index_col="cilia_id")  # ciliaIO
    
    m = pd.merge(d1.reset_index(), d2.reset_index(), on="cilia_id", suffixes=("_1", "_2"))

    m['fname'] = vid_name

    if not m.empty:
        # Show individual video stats
        calculate_metrics(m, label=f"Video: {vid_name}", include_length=False, include_frequency=True)
        thou_dfs.append(m)

# Process the Aggregated Group
if thou_dfs:
    combined_thou = pd.concat(thou_dfs, ignore_index=True)
    print("\n" + "="*50)
    calculate_metrics(combined_thou, label="TOTAL THOUVENIN (Aggregated)", include_length=False, include_frequency=True)
    print("="*50 + "\n")

    df_bbs2 = combined_thou[~combined_thou['fname'].str.contains('wt', na=False)]
    df_wt = combined_thou[~combined_thou['fname'].str.contains('bbs2', na=False)]

    plot_violin_and_utest(df_wt, 'frequency_1', "wt", "wt", '(Hz)')
    plot_violin_and_utest(df_bbs2, 'frequency_1', "bbs2", "bbs2", '(Hz)')

print("PROCESSING MANUAL VIDEOS\n")
manual_dfs = []
for m_path, q_path in manual_pairs:
    vid_name = os.path.basename(m_path).split("__")[0]
    d1 = pd.read_csv(m_path, index_col="cilia_id")
    d2 = pd.read_csv(q_path, index_col="cilia_id")
    
    m_manual = pd.merge(d1.reset_index(), d2.reset_index(), on="cilia_id", suffixes=("_1", "_2"))
    
    if not m_manual.empty:
        calculate_metrics(m_manual, label=f"Manual Video: {vid_name}", include_length=True, include_frequency=False)
        manual_dfs.append(m_manual)

if manual_dfs:
    combined_manual = pd.concat(manual_dfs, ignore_index=True)
    print("\n" + "="*50)
    calculate_metrics(combined_manual, label="TOTAL Manual (Aggregated)", include_length=True, include_frequency=False)
    print("="*50 + "\n")

# --- PLOTTING SECTION ---
if (combined_thou is not None and not combined_thou.empty) or (combined_manual is not None and not combined_manual.empty):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Frequency Errors (Thouvenin)
    if combined_thou is not None and not combined_thou.empty:
        # Using _1 as the reference/denominator to match MAPE logic
        abs_err_f = (combined_thou["frequency_1"] - combined_thou["frequency_2"]).abs()
        # Individual Absolute Percentage Error
        ape_f = (abs_err_f / combined_thou["frequency_1"]) * 100
        mabs_err_f = np.mean(abs_err_f)
        mape_f = np.mean(ape_f)
        
        axes[0, 0].hist(abs_err_f, bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Frequency: Absolute Error")
        axes[0, 0].set_xlabel("Absolute Error ($Hz$)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].axvline(mabs_err_f, label=f'Mean: {mabs_err_f:.2f} Hz', linestyle='--', color='black')
        axes[0, 0].legend()
        
        axes[0, 1].hist(ape_f, bins=50, color='salmon', edgecolor='black')
        axes[0, 1].set_title("Frequency: Absolute Percentage Error")
        axes[0, 1].set_xlabel("Error (%)")
        axes[0, 1].axvline(mape_f, label=f'MAPE: {mape_f:.2f}%', linestyle='--', color='black')
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

    # Length Errors (Manual)
    if combined_manual is not None and not combined_manual.empty:
        # Using _1 as the reference/denominator to match MAPE logic
        abs_err_l = (combined_manual["mean_cilia_length_1"] - combined_manual["mean_cilia_length_2"]).abs()
        mabs_err_l = np.mean(abs_err_l)
        # Individual Absolute Percentage Error
        ape_l = (abs_err_l / combined_manual["mean_cilia_length_1"]) * 100
        mape_l = np.mean(ape_l)
        
        axes[1, 0].hist(abs_err_l, bins=50, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title("Cilia Length: Absolute Error")
        axes[1, 0].set_xlabel("Absolute Error ($\mu m$)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].axvline(mabs_err_l, label=f'Mean: {mabs_err_l:.2f} $\mu m$', linestyle='--', color='black')
        axes[1, 0].legend()
        
        axes[1, 1].hist(ape_l, bins=50, color='plum', edgecolor='black')
        axes[1, 1].set_title("Cilia Length: Absolute Percentage Error")
        axes[1, 1].set_xlabel("Error (%)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].axvline(mape_l, label=f'MAPE: {mape_l:.2f}%', linestyle='--', color='black')
        axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Distribution of Absolute and Percentage Errors", fontsize=16)
    plt.show()