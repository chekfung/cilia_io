import os
import warnings
import pandas as pd

# matplotlib and seaborn visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
plt.rcParams['savefig.dpi'] = 300

# Stats
from scipy import stats
from statsmodels.stats.power import TTestIndPower 
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

# Warnings
import warnings
import logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning

'''
This file compiles all CSV files from different confocal microscope videos, 
and then starts doing statistical analysis between dorsal, ventral in WT and 
mutant zebrafish. It then generates violin plots, Mann Whitney U test between groups,
and Cohen's power analysis.

NOTE: DORSAL = Group 0, Ventral = Group 1 :)
'''

# === Data folder ===
folder = '../../data/test_cilia_io/'

SAVE_FIGURES = False
SAVE_DIRECTORY = "../../writing/figures"
alpha = 0.05
ALPHA = alpha
power = 0.8

run_arl_accumulation = True
CONTROL_NAME = "wt"
MUTANT_NAME = "bbs2"

# -----------
# Save Figures Create Directory
if SAVE_FIGURES:
    # Check if directory exists
    last_dir = os.path.basename(os.path.normpath(folder))
    figure_directory = os.path.join(SAVE_DIRECTORY, last_dir)
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

# Global maps for bbs2 and wt dataframes
bbs2_dfs = []
wt_dfs = []

# Power Analysis
power_analysis = TTestIndPower()

# Color Schemes for Dorsal and Ventral
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


# === Utils ===
def make_unique_cilia_id(df, prefix):
    df = df.copy()
    df['original_cilia_id'] = df['cilia_id']
    df['cilia_id'] = df['cilia_id'].apply(lambda x: f"{prefix}_{x}")
    return df

def significance_stars(p_val, alpha=ALPHA):
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

# === Plotting ===
def pca_per_cilium_2d_plots(df, identifier, use_other_axis=False, pca=None, scaler=None):
    data_full = df[feat_list].copy()
    groups = df['group']

    # Standardization
    if not use_other_axis:
        print("Creating new Standard Scaler")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_full)

        # Perform PCA 2D
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data_scaled)
        
    else:
        data_scaled = scaler.transform(data_full)
        pca_results = pca.transform(data_scaled)

    # Create a clean DataFrame for plotting
    pca_df = pd.DataFrame(
        data=pca_results, 
        columns=['PC1', 'PC2']
    )
    pca_df['Group'] = groups.values
    pca_df['Label'] = pca_df['Group'].map({0: "Dorsal", 1: "Ventral"})

    pca_df['Fish'] = df['fname'].values

    # 4. Visualization
    plt.figure(figsize=(10, 8))

    scatter = sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2',
        hue='Label',
        palette=MASTER_MAP,
        s=150, alpha=0.7, edgecolors='k'
    )

    # Explained Variance in Axis Labels
    exp_var = pca.explained_variance_ratio_ * 100
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([-4, 8])
    plt.ylim([-3, 4])
    plt.xlabel(f"PC1 ({exp_var[0]:.1f}% Variance)", fontsize=14)
    plt.ylabel(f"PC2 ({exp_var[1]:.1f}% Variance)", fontsize=14)
    plt.title(f"Full Feature PCA: ({identifier})\n(Dorsal vs Ventral)", fontsize=16)

    handles, labels = scatter.get_legend_handles_labels()
    
    # Filter to only keep Dorsal and Ventral entries
    leg_map = {l: h for h, l in zip(handles, labels) if l in ["Dorsal", "Ventral"]}
    
    final_handles = [leg_map["Dorsal"], leg_map["Ventral"]]
    final_labels = ["Dorsal", "Ventral"]

    # 3. Apply the clean legend
    plt.legend(final_handles, final_labels, title="Orientation", 
               fontsize=12, title_fontsize=13, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f'../writing/4_27_26_cilia_io_graphs/2d_pca_{identifier}.png', format='png', dpi=300)

    # Feature Loading Analysis (Who contributes most?)
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=feat_list
    )
    print("\n--- Top Contributors to PC1 ---")
    print(loadings['PC1'].sort_values(ascending=False).head(5))

    print("\n--- Top Contributors to PC2 ---")
    print(loadings['PC2'].sort_values(ascending=False).head(5))

    # 6. Statistical Test on PC1 (Same logic as your original)
    clean_groups = groups.reset_index(drop=True)
    clean_groups.index = clean_groups.index.astype(str)
    dist_matrix = pdist(data_scaled, metric='euclidean')
    sample_ids = [str(i) for i in range(len(data_scaled))]
    perm_results = permanova(DistanceMatrix(squareform(dist_matrix),ids=sample_ids), clean_groups, permutations=9999)

    print(f"PERMANOVA (Global): F-stat = {perm_results['test statistic']:.2f}, p = {perm_results['p-value']:.8f}")

    pc1_dorsal = pca_results[groups == 0, 0]
    pc1_ventral = pca_results[groups == 1, 0]
    pc2_dorsal = pca_results[groups == 0, 1]
    pc2_ventral = pca_results[groups == 1, 1]
    u_stat, p_val = stats.mannwhitneyu(pc1_dorsal, pc1_ventral)
    u2, p2 = stats.mannwhitneyu(pc2_dorsal, pc2_ventral)
    print(f"\nU-test on PC1: U = {u_stat}, p = {p_val:.2e}")
    print(f"\nU-test on PC2: U = {u2}, p = {p2:.2e}")

    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=feat_list
    )

    loadings['Magnitude'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)

    pca_table_horizontal = loadings.sort_values(by='PC1', ascending=False).T

    print("\n--- PCA Feature Contribution Table (Horizontal) ---")
    print(pca_table_horizontal.round(4))

    # Create a Horizontal Heatmap
    plt.figure(figsize=(16, 4)) 
    
    sns.heatmap(
        pca_table_horizontal.loc[['PC1', 'PC2']], 
        annot=True, 
        annot_kws={"size": 15, "weight": "bold"},
        cmap='flare', 
        center=0,
        cbar_kws={'label': 'Loading Weight', 'orientation': 'horizontal', 'pad': 0.3}
    )
    
    plt.title(f"Feature Contributions to Principal Components {identifier}", fontsize=16)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f'../writing/4_27_26_cilia_io_graphs/2d_pca_{identifier}_table.png', format='png', dpi=300)

    return pca, scaler

def pca_2d_plots(df, identifier, use_other_axis=False, pca=None, scaler=None):
    """
    Performs 2D PCA on per-fish medians and connects paired points (Dorsal-Ventral).
    """
    # Get fish medians
    df_fish = df.groupby(['fname', 'group'])[feat_list].median().reset_index()
    
    data_full = df_fish[feat_list].copy()
    groups = df_fish['group']
    
    # standarization + pca
    if not use_other_axis:
        print(f"\n--- Initializing PCA for {identifier} (Fish-Level) ---")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_full)
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data_scaled)
    else:
        print(f"\n--- Applying existing PCA to {identifier} ---")
        data_scaled = scaler.transform(data_full)
        pca_results = pca.transform(data_scaled)

    pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
    pca_df['Group'] = groups.values
    pca_df['Label'] = pca_df['Group'].map({0: "Dorsal", 1: "Ventral"})
    pca_df['Fish'] = df_fish['fname'].values

    plt.figure(figsize=(10, 8))
    
    # Draw lines connecting dorsal and ventral pairs together per fish
    for fish_id in pca_df['Fish'].unique():
        fish_data = pca_df[pca_df['Fish'] == fish_id]
        
        if len(fish_data) == 2:
            plt.plot(
                fish_data['PC1'].values, 
                fish_data['PC2'].values, 
                color='black', 
                linestyle='-', 
                linewidth=1.2, 
                alpha=0.2,      
                zorder=1       
            )

    # Each dot is one fish median
    scatter = sns.scatterplot(
        data=pca_df,
        x='PC1', y='PC2',
        hue='Label',
        style='Label',
        palette=MASTER_MAP,
        s=250, alpha=0.9, edgecolors='k', zorder=2  # High zorder to stay on top
    )

    exp_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({exp_var[0]:.1f}% Variance)", fontsize=14)
    plt.ylabel(f"PC2 ({exp_var[1]:.1f}% Variance)", fontsize=14)
    plt.title(f"Fish-Level PCA: {identifier}\n(Paired Dorsal-Ventral Medians)", fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f'../writing/4_27_26_cilia_io_graphs/2d_pca_{identifier}_fish.png', dpi=300)

    # 5. Permanova test
    sample_ids = [str(i) for i in range(len(data_scaled))]
    clean_groups = groups.reset_index(drop=True)
    clean_groups.index = clean_groups.index.astype(str)
    dist_matrix = pdist(data_scaled, metric='euclidean')
    sample_ids = [str(i) for i in range(len(data_scaled))]
    perm_results = permanova(DistanceMatrix(squareform(dist_matrix),ids=sample_ids), clean_groups, permutations=9999)
    
    print(f"PERMANOVA (Global): F-stat = {perm_results['test statistic']:.2f}, p = {perm_results['p-value']:.8f}")

    # Paired Statistics on individual PCs (Wilcoxon Signed-Rank)
    pc1_dorsal = pca_df[pca_df['Group'] == 0].sort_values('Fish')['PC1'].values
    pc1_ventral = pca_df[pca_df['Group'] == 1].sort_values('Fish')['PC1'].values
    
    w1, p1 = stats.wilcoxon(pc1_dorsal, pc1_ventral)
    print(f"Paired Wilcoxon on PC1: p = {p1:.2e}")

    pc2_dorsal = pca_df[pca_df['Group'] == 0].sort_values('Fish')['PC2'].values
    pc2_ventral = pca_df[pca_df['Group'] == 1].sort_values('Fish')['PC2'].values
    
    w2, p2 = stats.wilcoxon(pc2_dorsal, pc2_ventral)
    print(f"Paired Wilcoxon on PC2: p = {p2:.2e}")

    # Prep heatmap for plotting PCA components
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=feat_list
    )

    pca_table_horizontal = loadings.sort_values(by='PC1', ascending=False).T

    plt.figure(figsize=(16, 4)) 
    
    sns.heatmap(
        pca_table_horizontal.loc[['PC1', 'PC2']], 
        annot=True, 
        annot_kws={"size": 15, "weight": "bold"},
        cmap='flare', 
        center=0,
        cbar_kws={'label': 'Loading Weight', 'orientation': 'horizontal', 'pad': 0.3}
    )
    
    plt.title(f"Feature Contributions to Principal Components {identifier}", fontsize=16)
    plt.tight_layout()
    
    if SAVE_FIGURES:
        plt.savefig(f'../writing/4_27_26_cilia_io_graphs/2d_pca_{identifier}_loadings.png', dpi=300)

    return pca, scaler

def plot_violin_and_utest(df, feature, name, fname="", feat_unit=None):
    # Get dorsal ventral groupings
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

    # stat test per fish median
    g0_meds, g1_meds = paired_df[0], paired_df[1]
    stat, p_val = stats.wilcoxon(g0_meds, g1_meds)
    stars = significance_stars(p_val)
    
    # Perform for fish-level cv
    spread_df = fish_stats.pivot(index='fname', columns='group', values='std').dropna()
    if not spread_df.empty:
        _, p_var_spread = stats.wilcoxon(spread_df[0], spread_df[1])
        stars_var = significance_stars(p_var_spread)
    else:
        p_var_spread, stars_var = np.nan, "ns"

    # Perform for fish-level cv
    cv_df = fish_stats.pivot(index='fname', columns='group', values='cv').dropna()
    if not cv_df.empty:
        _, p_cv = stats.wilcoxon(cv_df[0], cv_df[1])
        stars_cv = significance_stars(p_cv)
    else:
        p_cv = np.nan
        stars_cv = "ns"

    # Power analysis on cilia (deprecated)
    n1, n2 = len(g0), len(g1)
    mean_diff = np.mean(g1) - np.mean(g0)
    s1, s2 = np.std(g0, ddof=1), np.std(g1, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    cohen_d_cilia = mean_diff / (pooled_std if pooled_std != 0 else 1)
    required_n_cilia = power_analysis.solve_power(effect_size=max(abs(cohen_d_cilia), 0.0001),
                                                 alpha=0.05, power=0.8, ratio=n2/n1)

    # Power analysis using fish medians
    effect_size_fish, required_n_fish = calculate_nested_power(df, feature)
    effect_size_fish = float(effect_size_fish)
    required_n_fish = float(np.array(required_n_fish).item())

    # Iteratively plot all points fit on the plots
    point_size = 4.5
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
                          palette="tab20", dodge=False, size=point_size, ax=ax, alpha=0.3, zorder=2)

            # Fish Medians (Diamonds)
            sns.stripplot(x="group", y=feature, data=plot_fish_meds, hue="fname",
                          palette="tab20", marker="D", size=7, jitter=0.1, 
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

    # Additional Plot Updates
    if ax.get_legend() is not None: ax.get_legend().remove()
    plt.xticks([0, 1], ["Dorsal", "Ventral"])
    unit_str = f" ({feat_unit})" if feat_unit else ""
    plt.title(f"{feature.replace('_',' ').title()} {name}", fontsize=11)
    plt.ylabel(feature.replace("_", " ").title() + unit_str)
    plt.xlabel("")
    sns.despine()

    # Graph stars if significant
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

    if SAVE_FIGURES:
        plt.savefig(os.path.join(figure_directory, f"{name}_{feature}_paired.png"), 
                    dpi=300, bbox_inches='tight')

# No support for superplots right now. (Deprecated)
def plot_violin_arl_accumulation_two_groups(df1, df, name1, name, feature, feature_unit):
    group_map = {0: "Dorsal", 1: "Ventral"}
    results = {}

    # Split mutant into accumulation vs not
    arl_accumulated = df[df['arl13b_accumulation'] == 1]
    arl_not_accumulated = df[df['arl13b_accumulation'] == 0]

    for g in [0, 1]:
        # Get values
        g_accum = arl_accumulated[arl_accumulated["group"]==g][feature].dropna()
        g_notaccum = arl_not_accumulated[arl_not_accumulated["group"]==g][feature].dropna()
        g_wt = df1[df1["group"]==g][feature].dropna()

        if g_wt.empty:
            print(f"Warning: {CONTROL_NAME} {feature}, group {group_map[g]} empty, skipping")
            continue

        # Build plotting dataframe
        data = []
        data.append(pd.DataFrame({"value": g_wt, "dataset": name1}))
        if not g_accum.empty:
            data.append(pd.DataFrame({"value": g_accum, "dataset": f"{name} (accum)"}))
        if not g_notaccum.empty:
            data.append(pd.DataFrame({"value": g_notaccum, "dataset": f"{name} (not accum)"}))
        
        data = pd.concat(data, ignore_index=True)

        # Palette
        if g == 0:
            new_palette = [cv2_to_rgb(DORSAL_COLOR)] * data["dataset"].nunique()
        else:
            new_palette = [cv2_to_rgb(VENTRAL_COLOR)] * data["dataset"].nunique()

        # Violin plot
        plt.figure(figsize=(4.5, 4))
        ax = sns.violinplot(
            x="dataset", y="value", data=data, hue="dataset",
            palette=new_palette, inner="box", cut=0, legend=False
        )

        # Hatches
        hatch_patterns = ["///", "...", "\\\\\\"]
        for i, collection in enumerate(ax.collections):
            if i == 0:  # skip first
                continue
            hatch = hatch_patterns[(i // 2) % len(hatch_patterns)]
            collection.set_hatch(hatch)
            collection.set_edgecolor("black")

        plt.title(f"{feature.replace('_',' ').title()} {group_map[g]}")
        plt.xlabel("")

        if feature_unit != None:
            plt.ylabel(feature.replace("_", " ").title() + " "+feature_unit)
        else:
            plt.ylabel(feature.replace("_", " ").title())

        sns.despine()
        plt.tight_layout()

        # Significance testing vs WT
        group_results = {}
        comparisons = []
        if not g_accum.empty:
            comparisons.append((f"accum_vs_{CONTROL_NAME}", g_accum, g_wt))
        if not g_notaccum.empty:
            comparisons.append((f"notaccum_vs_{CONTROL_NAME}", g_notaccum, g_wt))
        if not g_accum.empty and not g_notaccum.empty:
            comparisons.append(("accum_vs_not_accum", g_accum, g_notaccum))

        y_max = data["value"].max()
        y_min = data["value"].min()
        y, h = y_max + 0.1*(y_max-y_min), 0.07*(y_max-y_min)

        for i, (label, mutant_vals, wt_vals) in enumerate(comparisons):
            u_stat, p_val = stats.mannwhitneyu(mutant_vals, wt_vals, alternative='two-sided')
            stars = significance_stars(p_val)

            # Effect size
            n1, n2 = len(mutant_vals), len(wt_vals)
            mean_diff = mutant_vals.mean() - wt_vals.mean()
            s1, s2 = mutant_vals.std(ddof=1), wt_vals.std(ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
            ratio = n2 / n1 if n1 > 0 else np.nan
            required_n = np.nan
            if not np.isnan(cohen_d):
                try:
                    required_n = power_analysis.solve_power(
                        effect_size=abs(cohen_d), alpha=alpha, power=power, ratio=ratio
                    )
                except Exception:
                    pass

            # Print
            print(f"{label}: {name} vs {name1}, {feature}, {group_map[g]} "
                  f"(Required N/group: {required_n:.1f}, n={n1},{n2}) "
                  f"U={u_stat:.4f}, p={p_val:.4e}, stars={stars}")

            # Annotate plot with stars
            if stars != "ns":
                if i == 0:
                    x1, x2 = 0, 1
                elif i == 1:
                    x1, x2 = 0, 2
                elif i == 2:
                    x1, x2 = 1, 2

                ax.plot([x1, x1, x2, x2],
                    [y, y+h, y+h, y], lw=1.5, c="k")

                ax.text((x1 + x2)/2, y+h, stars,
                        ha="center", va="bottom", fontsize=12)
                
                y += 0.2*(y_max-y_min)  # stack bars vertically if multiple

            group_results[label] = {
                "u": u_stat, "p": p_val, "cohen_d": cohen_d,
                "required_n": required_n, "n1": n1, "n2": n2
            }
            
        if SAVE_FIGURES:
            filename = os.path.join(figure_directory, f"arl_accum_compare_{feature}_{group_map[g]}_comparison.png")
            plt.savefig(filename, dpi=400)

        results[g] = group_results

    return results

def plot_violin_and_utest_comparison_by_group(df0, df1, name0, name1, feature, group_col="group", feat_unit=None):
    group_map = {0: "Dorsal", 1: "Ventral"}
    groups = sorted(pd.concat([df0[group_col], df1[group_col]]).dropna().unique())
    results = {}

    for g in groups:
        # Filter data for this specific group (e.g., just Dorsal)
        sub0 = df0[df0[group_col] == g].copy()
        sub1 = df1[df1[group_col] == g].copy()

        if sub0.empty or sub1.empty:
            continue

        sub0['dataset_label'], sub0['dataset_bin'] = name0, 0
        sub1['dataset_label'], sub1['dataset_bin'] = name1, 1
        
        data = pd.concat([sub0, sub1], ignore_index=True)
        data['value'] = data[feature]

        # Calculate Median, SD, and CV for each fish
        fish_stats = data.groupby(['fname', 'dataset_label'])['value'].agg(['median', 'std', 'mean']).reset_index()
        fish_stats['cv'] = fish_stats['std'] / fish_stats['mean']

        # Median Comparison, use mann whitney since not pairwise 
        meds0 = fish_stats[fish_stats['dataset_label'] == name0]['median']
        meds1 = fish_stats[fish_stats['dataset_label'] == name1]['median']
        u_stat_med, p_val_med = stats.mannwhitneyu(meds0, meds1, alternative='two-sided')
        stars_med = significance_stars(p_val_med)

        # SD Comparison  for each fish
        sd0 = fish_stats[fish_stats['dataset_label'] == name0]['std'].dropna()
        sd1 = fish_stats[fish_stats['dataset_label'] == name1]['std'].dropna()
        _, p_val_sd = stats.mannwhitneyu(sd0, sd1, alternative='two-sided')
        stars_sd = significance_stars(p_val_sd)

        # CV Comparison for each fish
        cv0 = fish_stats[fish_stats['dataset_label'] == name0]['cv'].dropna()
        cv1 = fish_stats[fish_stats['dataset_label'] == name1]['cv'].dropna()
        _, p_val_cv = stats.mannwhitneyu(cv0, cv1, alternative='two-sided')
        stars_cv = significance_stars(p_val_cv)

        coeff = meds1.mean() - meds0.mean()

        # Iteratively plot until all swarmplot points fit
        point_size = 4.0
        iteration, max_iterations = 0, 50
        
        while iteration < max_iterations:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                plt.figure(figsize=(4, 5))
                ax = plt.gca()
                
                # background violin from cilia data
                new_palette = [cv2_to_rgb(DORSAL_COLOR if g == 0 else VENTRAL_COLOR)] * 2
                sns.violinplot(x="dataset_label", y="value", data=data, hue="dataset_label",
                               palette=new_palette, inner=None, cut=0, alpha=0.3, zorder=1, legend=False)

                # Plot individual cilia, colored based on fish
                sns.swarmplot(x="dataset_label", y="value", data=data, hue="fname",
                              palette="tab20", dodge=False, size=point_size, ax=ax, alpha=0.4, zorder=2)

                # Fish medians for each feature, colored by fish
                sns.stripplot(x="dataset_label", y="median", data=fish_stats, hue="fname",
                              palette="tab20", color="black", marker="D", size=7, jitter=0.1, 
                              edgecolor="black", linewidth=0.8,ax=ax, zorder=15,alpha=0.8)

                # box plot
                sns.boxplot(
                        x="dataset_label", y='value', data=data,
                        ax=ax,
                        width=0.07,             # Very narrow to look like a violin inner
                        showfliers=False,      
                        boxprops={'facecolor': '#333333', 'zorder': 10, 'edgecolor': 'none', 'alpha':0.9}, 
                        whiskerprops={'color': '#333333', 'linewidth': 1, 'zorder': 10, 'alpha':0.9},
                        medianprops={'color': 'white', 'linewidth': 1.3, 'zorder': 11, 'alpha':0.9},
                        capprops={'linewidth': 0}, # Fixed: This is how you hide the caps
                    )

                warning_triggered = any("points cannot be placed" in str(warning.message) for warning in w)
                if warning_triggered and iteration < max_iterations - 1:
                    plt.close(); point_size *= 0.8; iteration += 1; continue
                else:
                    break

        # Formatting
        if ax.get_legend() is not None: ax.get_legend().remove()
        unit_str = f" ({feat_unit})" if feat_unit else ""
        plt.title(f"{feature.replace('_',' ').title()} ({group_map[g]})\n{name0} vs {name1}", fontsize=12)
        plt.ylabel(feature.replace("_", " ").title() + unit_str)
        plt.xlabel("")
        sns.despine()

        # Annotation (Using Median stars for the plot)
        y_max, y_min = data["value"].max(), data["value"].min()
        y, h = y_max + 0.05*(y_max-y_min), 0.03*(y_max-y_min)
        if stars_med != "ns":
            ax.text(0.5, y+h, stars_med, ha="center", va="bottom", fontsize=14, fontweight='bold')
            ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c="k")

        print(f"--- {group_map[g]}: {feature} ---")
        print(f"Genotype Effect (Medians) : p={p_val_med:.4e} ({stars_med})")
        print(f"Variance Effect (SD)      : p={p_val_sd:.4e} ({stars_sd})")
        print(f"Precision Effect (CV)     : p={p_val_cv:.4e} ({stars_cv})")
        print(f"Mean Diff: {coeff:.4f} (if > 0, bbs2 higher)")

        if SAVE_FIGURES:
            filename = os.path.join(figure_directory, f"comparison_{name0}_{name1}_{feature}_{group_map[g]}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        results[g] = {"coeff": coeff, "p_med": p_val_med, "p_sd": p_val_sd, "p_cv": p_val_cv}
        
    return results

# ----------------------------------
# START OF THE MAIN SCRIPT

# === Data loading ===
num_bbs2 = 0
num_wt = 0

for fname in os.listdir(folder):
    if "skel" in fname and fname.endswith(".csv"):
        full_path = os.path.join(folder, fname)
        print(full_path)
        df = pd.read_csv(full_path)
        if "cilia_id" not in df.columns:
            print(f"Warning: cilia_id missing in {fname}, skipping")
            continue
        prefix = os.path.splitext(fname)[0]
        df = make_unique_cilia_id(df, prefix)

        # Add in the filename so that we can split the fish easier later :)
        df['fname'] = fname

        if MUTANT_NAME in fname.lower():
            bbs2_dfs.append(df); num_bbs2+=1
        elif CONTROL_NAME in fname.lower():
            wt_dfs.append(df); num_wt+=1

# Concatenate all the different CSV files together
bbs2_all = pd.concat(bbs2_dfs, ignore_index=True) if bbs2_dfs else pd.DataFrame()
wt_all = pd.concat(wt_dfs, ignore_index=True) if wt_dfs else pd.DataFrame()

print(f"Loaded {len(bbs2_all)} rows from {MUTANT_NAME} files")
print(f"Loaded {len(wt_all)} rows from {CONTROL_NAME} files")

try:    
    print(f"{MUTANT_NAME} ARL Accumulation: {bbs2_all['arl13b_accumulation'].sum()}, No Accumulation: {len(bbs2_all) - bbs2_all['arl13b_accumulation'].sum()}")
except:
    print("No ARL Accumulation Data")
    run_arl_accumulation = False

# Save filtered CSVs
if not bbs2_all.empty:
    bbs2_all.to_csv(os.path.join(folder,f"1_{MUTANT_NAME}_all_filtered.csv"), index=False)
if not wt_all.empty:
    wt_all.to_csv(os.path.join(folder,f"1_{CONTROL_NAME}_all_filtered.csv"), index=False)

feat_list = ["frequency", "centroid_frequency","mean_amplitude","max_amplitude",
             "entropy", "eccentricity","average_speed","mean_cilia_length","mean_cilia_area",
             "mean_straightness_flipped","std_straightness_flipped","max_straightness_flipped","min_straightness_flipped"]

feat_unit = ["(Hz)", "(Hz)", "($\mu$m)", "($\mu$m)", None, None, "($\mu$m/sec)", "($\mu$m)", "($\mu$m$^2$)",
             None, None, None, None]

# === Analyses ===
if not bbs2_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_utest(bbs2_all, feat, MUTANT_NAME, fname=MUTANT_NAME, feat_unit=feat_unit[i])

if not wt_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_utest(wt_all, feat, CONTROL_NAME, fname=CONTROL_NAME, feat_unit=feat_unit[i])

if not bbs2_all.empty and not wt_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_utest_comparison_by_group(wt_all, bbs2_all, CONTROL_NAME, MUTANT_NAME, feat, feat_unit=feat_unit[i])

#     # Deprecated  
#     if run_arl_accumulation:
#         for i, feat in enumerate(feat_list):
#             plot_violin_arl_accumulation_two_groups(wt_all, bbs2_all, CONTROL_NAME, MUTANT_NAME, feat, feat_unit[i])

# ------------------
# 3D Plots and 2d PCA below :)

if not wt_all.empty:
    # Plot wt 3D plot of all cilia
    count_0 = (wt_all['group'] == 0).sum()
    count_1 = (wt_all['group'] == 1).sum()

    fig = plt.figure()
    x = wt_all['entropy']
    y = wt_all["mean_amplitude"]
    z = wt_all["frequency"]
    groups = wt_all['group']

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=groups, cmap=mcolors.ListedColormap(PALETTE), s=200, alpha=0.6,edgecolors='k')
    ax.set_title(f"{CONTROL_NAME} Groups (Dorsal Vs. Ventral) (N={num_wt}) Dorsal: {count_0} Ventral: {count_1}", fontsize=20)

    ax.set_xlabel("Entropy", fontsize=18, labelpad=15)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=18, labelpad=15)
    ax.set_zlabel("Frequency (Hz)", fontsize=18, labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # Manually assign legend using dummy points
    cmap = mcolors.ListedColormap(PALETTE)
    group_to_label = {0: "Dorsal", 1: "Ventral"}

    for group_val, label in group_to_label.items():
        ax.scatter([], [], [],  # Invisible point
                color=cmap(group_val / 1), 
                label=label,
                s=60, alpha=0.6)

    ax.legend(fontsize=14, markerscale=1.5)

    # Combine the 3 lists into a 3D dataset
    data_3d = np.array([x, y, z]).T

    # Column-wise Normalization
    scaler = StandardScaler()
    data_3d = scaler.fit_transform(data_3d)

    # Perform PCA to reduce to 1 dimension
    pca = PCA(n_components=1)  
    pca.fit(data_3d)          
    data_1d = pca.transform(data_3d) #
    print("entropy , max_amplitude , frequency")
    print(pca.components_)
    exp_var = pca.explained_variance_ratio_ * 100
    print(f"Variance: {exp_var}%")

    # Visualize the 1D data with coloring

    # Get unique categories
    unique_categories = np.unique(groups)
    colors = mcolors.ListedColormap(PALETTE)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    bins = 20  # adjust bin count if needed
    for i, category in enumerate(unique_categories):
        indices = np.where(groups == category)[0]
        plt.hist(
            data_1d[indices],
            bins=bins,
            alpha=0.6,
            color=colors(i),
            label=str(category)
        )

    plt.title('PCA Dimension 1 Histogram by Group', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Number of Cilia', fontsize=14)
    plt.legend(['Dorsal', 'Ventral'], fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Logistic regression with statsmodels
    X = sm.add_constant(data_1d)  # add intercept
    logit_model = sm.Logit(groups, X)
    result = logit_model.fit()

    print(result.summary())

    # U-test for PC1 between groups ---
    pc1_dorsal = data_1d[groups == 0].ravel()
    pc1_ventral = data_1d[groups == 1].ravel()
    u_stat, p_val = stats.mannwhitneyu(pc1_dorsal, pc1_ventral, alternative='two-sided')

    print(f"U-test: u = {u_stat:.6f}, p = {p_val:.8e}")

    group0 = data_1d[groups == 0]
    group1 = data_1d[groups == 1]

    # Calculate Cohen's d for power analysis
    n1, n2 = len(group0), len(group1)
    mean_diff = np.mean(group0) - np.mean(group1)
    s1, s2 = np.std(group0, ddof=1), np.std(group1, ddof=1)
    pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    cohen_d = mean_diff / pooled_std

    # Calculate required sample size per group for 80% power
    ratio = n2 / n1
    required_n = power_analysis.solve_power(effect_size=abs(cohen_d), alpha=alpha, power=power, ratio=ratio)

    print(f"Power Analysis for PCA Decomposition. REQUIRE: {required_n:.2f} N/GROUP ; Current Numbers (Dorsal, Ventral): {n1}, {n2}")

if not bbs2_all.empty:
    # Plot bbs2 3D plot for all cilia
    count_0 = (bbs2_all['group'] == 0).sum()
    count_1 = (bbs2_all['group'] == 1).sum()

    fig = plt.figure()

    x = bbs2_all['entropy']
    y = bbs2_all["mean_amplitude"]
    z = bbs2_all["frequency"]
    groups = bbs2_all['group']

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=groups, cmap=mcolors.ListedColormap(PALETTE), s=200, alpha=0.6,edgecolors='k')
    ax.set_title(f"{MUTANT_NAME} Groups (Dorsal Vs. Ventral) (N={num_bbs2}) Dorsal: {count_0} Ventral: {count_1}", fontsize=20)

    ax.set_xlabel("Entropy", fontsize=18, labelpad=15)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=18, labelpad=15)
    ax.set_zlabel("Frequency (Hz)", fontsize=18, labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # Manually assign legend using dummy points
    cmap = mcolors.ListedColormap(PALETTE)
    group_to_label = {0: "Dorsal", 1: "Ventral"}

    for group_val, label in group_to_label.items():
        ax.scatter([], [], [],  
                color=cmap(group_val / 1),  
                label=label,
                s=60, alpha=0.6)

    ax.legend(fontsize=14, markerscale=1.5)

    # Combine the 3 lists into a 3D dataset
    data_3d = np.array([x, y, z]).T

    # Column-wise Normalization
    scaler = StandardScaler()
    data_3d = scaler.fit_transform(data_3d)

    # Perform PCA to reduce to 1 dimension
    pca = PCA(n_components=1)  
    pca.fit(data_3d)          
    data_1d = pca.transform(data_3d) 
    print("entropy , max_amplitude , frequency")
    print(pca.components_)
    exp_var = pca.explained_variance_ratio_ * 100
    print(f"Variance: {exp_var}%")

    # Visualize the 1D data with coloring

    # Get unique categories
    unique_categories = np.unique(groups)
    colors = mcolors.ListedColormap(PALETTE)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    bins = 20  # adjust bin count if needed
    for i, category in enumerate(unique_categories):
        indices = np.where(groups == category)[0]
        plt.hist(
            data_1d[indices],
            bins=bins,
            alpha=0.6,
            color=colors(i),
            label=str(category)
        )

    plt.title(f'PCA Dimension 1 Histogram by Group ({MUTANT_NAME})', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Number of Cilia', fontsize=14)
    plt.legend(['Dorsal', 'Ventral'], fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Logistic regression with statsmodels
    X = sm.add_constant(data_1d)  # add intercept
    logit_model = sm.Logit(groups, X)
    result = logit_model.fit()

    print(f"{MUTANT_NAME} Logistic Regression PCA")
    print(result.summary())

    # --- U-test for PC1 between groups ---
    pc1_dorsal = data_1d[groups == 0].ravel()
    pc1_ventral = data_1d[groups == 1].ravel()
    u_stat, p_val = stats.mannwhitneyu(pc1_dorsal, pc1_ventral, alternative='two-sided')

    print(f"{MUTANT_NAME} PCA U-test: u = {u_stat:.6f}, p = {p_val:.8e}")

    group0 = data_1d[groups == 0]
    group1 = data_1d[groups == 1]

    # Calculate Cohen's d for power analysis
    n1, n2 = len(group0), len(group1)
    mean_diff = np.mean(group0) - np.mean(group1)
    s1, s2 = np.std(group0, ddof=1), np.std(group1, ddof=1)
    pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    cohen_d = mean_diff / pooled_std

    # Calculate required sample size per group for 80% power
    ratio = n2 / n1
    required_n = power_analysis.solve_power(effect_size=abs(cohen_d), alpha=alpha, power=power, ratio=ratio)

    print(f"Power Analysis for PCA Decomposition. REQUIRE: {required_n:.2f} N/GROUP ; Current Numbers (Dorsal, Ventral): {n1}, {n2}")


if not wt_all.empty and not bbs2_all.empty:
    # Plot WT, bbs2 together on 3D Plot

     # Filter data for Group 0 (Dorsal)
    wt_dorsal = wt_all[wt_all['group'] == 0]
    bbs2_dorsal = bbs2_all[bbs2_all['group'] == 0]

    # Setup Colors
    c_wt = '#E66101' 
    c_bbs2 = '#1F77B4'

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot WT Dorsal
    ax1.scatter(wt_dorsal['entropy'], 
                wt_dorsal['mean_amplitude'], 
                wt_dorsal['frequency'], 
                color=c_wt, label=f'{CONTROL_NAME} (Dorsal)', 
                s=200, alpha=0.6, edgecolors='k')

    # Plot BBS2 Dorsal
    ax1.scatter(bbs2_dorsal['entropy'], 
                bbs2_dorsal['mean_amplitude'], 
                bbs2_dorsal['frequency'], 
                color=c_bbs2, label='BBS2 (Dorsal)', 
                s=200, alpha=0.6, edgecolors='k')

    # Labels & Formatting
    ax1.set_title(f"Dorsal Comparison: {CONTROL_NAME} vs BBS2", fontsize=18)
    ax1.set_xlabel("Entropy", fontsize=14, labelpad=10)
    ax1.set_ylabel(r"Mean Amplitude ($\mu$m)", fontsize=14, labelpad=10)
    ax1.set_zlabel("Frequency (Hz)", fontsize=14, labelpad=10)
    ax1.legend(fontsize=12)

    # For Ventral
    wt_ventral = wt_all[wt_all['group'] == 1]
    bbs2_ventral = bbs2_all[bbs2_all['group'] == 1]

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot WT Ventral
    ax2.scatter(wt_ventral['entropy'], 
                wt_ventral['mean_amplitude'], 
                wt_ventral['frequency'], 
                color=c_wt, label=f'{CONTROL_NAME} (Ventral)', 
                s=200, alpha=0.6, edgecolors='k')

    # Plot BBS2 Ventral
    ax2.scatter(bbs2_ventral['entropy'], 
                bbs2_ventral['mean_amplitude'], 
                bbs2_ventral['frequency'], 
                color=c_bbs2, label='BBS2 (Ventral)', 
                s=200, alpha=0.6, edgecolors='k')

    # Labels & Formatting
    ax2.set_title(f"Ventral Comparison: {CONTROL_NAME} vs BBS2", fontsize=18)
    ax2.set_xlabel("Entropy", fontsize=14, labelpad=10)
    ax2.set_ylabel(r"Mean Amplitude ($\mu$m)", fontsize=14, labelpad=10)
    ax2.set_zlabel("Frequency (Hz)", fontsize=14, labelpad=10)
    ax2.legend(fontsize=12)

# ------------------- 
# 2D PCA Plot

feat_list = ["frequency", "max_amplitude",
             "entropy", "average_speed",
             "mean_straightness_flipped","std_straightness_flipped"]

feat_unit = ["(Hz)", "($\mu$m)", None,  "($\mu$m/sec)",
             None, None]

SAME_AXIS = True    # Project bbs2 onto the same axis as wt for comparison

if not wt_all.empty:
    print("2D WT")
    wt_axis_pca, wt_scaler = pca_2d_plots(wt_all, "wt")

if not bbs2_all.empty:
    print("2D BBS2")

    if SAME_AXIS:
        pca_2d_plots(bbs2_all, "bbs2", use_other_axis = SAME_AXIS, pca=wt_axis_pca, scaler=wt_scaler)
    else:
        pca_2d_plots(bbs2_all, "bbs2")

# Project bbs2 onto wt (cilia-level)
if not wt_all.empty and not bbs2_all.empty:
    wt_data = wt_all[feat_list].copy()
    
    scaler = StandardScaler()
    scaler.fit(wt_data) # Establish WT scaling
    wt_scaled = scaler.transform(wt_data)
    
    pca = PCA(n_components=2)
    pca.fit(wt_scaled) # Establish WT PCA directions
    
    # Project WT and bbs2 into its space
    wt_pca = pca.transform(wt_scaled)

    mutant_data = bbs2_all[feat_list].copy()
    mutant_scaled = scaler.transform(mutant_data)
    mutant_pca = pca.transform(mutant_scaled)
    
    df_wt = pd.DataFrame(wt_pca, columns=['PC1', 'PC2'])
    df_wt['Genotype'] = 'Control (WT)'
    df_wt['Orientation'] = wt_all['group'].map({0: "Dorsal", 1: "Ventral"}).values
    
    df_mut = pd.DataFrame(mutant_pca, columns=['PC1', 'PC2'])
    df_mut['Genotype'] = 'Mutant (BBS2)'
    df_mut['Orientation'] = bbs2_all['group'].map({0: "Dorsal", 1: "Ventral"}).values
    
    combined_pca = pd.concat([df_wt, df_mut])

    plt.figure(figsize=(12, 8))
    
    # Hue handles the Genotype (Color), Style handles the Orientation (Marker)
    sns.scatterplot(
        data=combined_pca,
        x='PC1', y='PC2',
        hue='Genotype',
        style='Orientation',
        palette={'Control (WT)': 'blue', 'Mutant (BBS2)': 'red'},
        s=150, alpha=0.6, edgecolors='k'
    )
    
    plt.title(f"Mutant projected onto Control PCA Space", fontsize=16)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

if not wt_all.empty and not bbs2_all.empty:
    # PCA with WT and BBS2 on same plot projected (medians)

    # --- medians first ---
    wt_fish = wt_all.groupby(['fname', 'group'])[feat_list].median().reset_index()
    mut_fish = bbs2_all.groupby(['fname', 'group'])[feat_list].median().reset_index()

    #  fit pca to wt
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    
    wt_data = wt_fish[feat_list]
    wt_scaled = scaler.fit_transform(wt_data)
    wt_pca_res = pca.fit_transform(wt_scaled)
    
    # bbs2 to wt pca space
    mut_data = mut_fish[feat_list]
    mut_scaled = scaler.transform(mut_data)
    mut_pca_res = pca.transform(mut_scaled)

    # Build DataFrames
    df_wt = pd.DataFrame(wt_pca_res, columns=['PC1', 'PC2'])
    df_wt['Genotype'], df_wt['Fish'], df_wt['Group'] = 'WT', wt_fish['fname'], wt_fish['group']
    
    df_mut = pd.DataFrame(mut_pca_res, columns=['PC1', 'PC2'])
    df_mut['Genotype'], df_mut['Fish'], df_mut['Group'] = 'BBS2', mut_fish['fname'], mut_fish['group']
    
    combined_pca = pd.concat([df_wt, df_mut])
    combined_pca['Label'] = combined_pca['Group'].map({0: "Dorsal", 1: "Ventral"})

    plt.figure(figsize=(10, 8))
    
    # Draw connection lines between dorsal, ventral for each fish
    for geno, color, style in [('WT', 'gray', '-'), ('BBS2', 'gray', '--')]:
        sub = combined_pca[combined_pca['Genotype'] == geno]
        for fish in sub['Fish'].unique():
            pair = sub[sub['Fish'] == fish].sort_values('Group')
            if len(pair) == 2:
                plt.plot(pair['PC1'].values, pair['PC2'].values, color=color, alpha=0.4, linestyle=style, zorder=1)

    sns.scatterplot(data=combined_pca, x='PC1', y='PC2', hue='Label', style='Genotype',
                    palette=MASTER_MAP, markers={'WT': 'o', 'BBS2': 'X'}, 
                    s=200, edgecolors='k', alpha=0.8, zorder=2)
    
    # Formatting
    plt.legend(labelspacing=1.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.title(f"WT, BBS2 PCA Projection")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- pc group coordinate shift comparisons ---
    # PC1 (Entropy/Straightness) Comparisons
    d_wt_pc1 = combined_pca[(combined_pca['Genotype'] == 'WT') & (combined_pca['Group'] == 0)]['PC1']
    d_mu_pc1 = combined_pca[(combined_pca['Genotype'] == 'BBS2') & (combined_pca['Group'] == 0)]['PC1']
    v_wt_pc1 = combined_pca[(combined_pca['Genotype'] == 'WT') & (combined_pca['Group'] == 1)]['PC1']
    v_mu_pc1 = combined_pca[(combined_pca['Genotype'] == 'BBS2') & (combined_pca['Group'] == 1)]['PC1']

    # PC2 (Frequency/Speed) Comparisons
    d_wt_pc2 = combined_pca[(combined_pca['Genotype'] == 'WT') & (combined_pca['Group'] == 0)]['PC2']
    d_mu_pc2 = combined_pca[(combined_pca['Genotype'] == 'BBS2') & (combined_pca['Group'] == 0)]['PC2']
    v_wt_pc2 = combined_pca[(combined_pca['Genotype'] == 'WT') & (combined_pca['Group'] == 1)]['PC2']
    v_mu_pc2 = combined_pca[(combined_pca['Genotype'] == 'BBS2') & (combined_pca['Group'] == 1)]['PC2']

    # Run the U-Tests
    stats_summary = [
        ("Dorsal PC1", stats.mannwhitneyu(d_wt_pc1, d_mu_pc1)),
        ("Ventral PC1", stats.mannwhitneyu(v_wt_pc1, v_mu_pc1)),
        ("Dorsal PC2", stats.mannwhitneyu(d_wt_pc2, d_mu_pc2)),
        ("Ventral PC2", stats.mannwhitneyu(v_wt_pc2, v_mu_pc2))
    ]

    print("\n--- POSITIONAL SIGNIFICANCE (U-TEST) ---")
    for label, (u, p) in stats_summary:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{label:25} | p = {p:.4f} ({sig})")


# -------------------
# Label IDs plus fish based on filename
# Used for determining which fish IDs for which cilia

if not wt_all.empty:
    x = wt_all['entropy']
    y = wt_all["mean_amplitude"]
    z = wt_all["frequency"]
    ids = wt_all['original_cilia_id']
    fnames = wt_all['fname']
    groups = wt_all['group']

    # Create a unique color for each filename
    unique_fnames = fnames.unique()
    fname_to_idx = {name: i for i, name in enumerate(unique_fnames)}
    file_cmap = plt.get_cmap('tab20') 

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting loop to handle markers (Dorsal vs Ventral)
    for group_val, marker, label in zip([0, 1], ['o', 'D'], ['Dorsal', 'Ventral']):
        mask = groups == group_val
        
        # Scatter plot for this group
        curr_colors = fnames[mask].map(fname_to_idx)
        scatter = ax.scatter(x[mask], y[mask], z[mask], 
                            c=curr_colors, 
                            cmap=file_cmap,
                            vmin=0, vmax=len(unique_fnames)-1,
                            marker=marker, 
                            s=250, alpha=0.6, edgecolors='k')

    # Add ID labels to the center of every point
    for i in range(len(wt_all)):
        ax.text(x.iloc[i], y.iloc[i], z.iloc[i], 
                str(ids.iloc[i]),
                color='black', fontsize=8, fontweight='bold',
                ha='center', va='center')

    # Create the Legends
    # Legend A: Region (Shape)
    region_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markeredgecolor='k', markersize=10, label='Dorsal'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                markeredgecolor='k', markersize=10, label='Ventral')
    ]

    # Legend B: File Names (Color)
    file_handles = []
    for name in unique_fnames:
        color = file_cmap(fname_to_idx[name] / max(1, len(unique_fnames)-1))
        handle = plt.Line2D([0], [0], marker='s', color='w', 
                            markerfacecolor=color, markersize=10, label=name)
        file_handles.append(handle)

    # Add Region Legend
    first_legend = ax.legend(handles=region_handles, title="Region", 
                            loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.add_artist(first_legend)

    ax.legend(handles=file_handles, title="File Names", 
            loc='upper left', bbox_to_anchor=(1.05, 0.85), ncol=1)

    # Final Styling
    ax.set_title(f"WT Distribution", fontsize=20)
    ax.set_xlabel("Entropy", fontsize=16, labelpad=10)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=16, labelpad=10)
    ax.set_zlabel("Frequency (Hz)", fontsize=16, labelpad=10)

    plt.tight_layout()


if not bbs2_all.empty:
    x = bbs2_all['entropy']
    y = bbs2_all["mean_amplitude"]
    z = bbs2_all["frequency"]
    ids = bbs2_all['original_cilia_id']
    fnames = bbs2_all['fname']
    groups = bbs2_all['group']

    # Create a unique color for each filename
    unique_fnames = fnames.unique()
    fname_to_idx = {name: i for i, name in enumerate(unique_fnames)}
    file_cmap = plt.get_cmap('tab20') 

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting loop to handle markers (Dorsal vs Ventral)
    for group_val, marker, label in zip([0, 1], ['o', 'D'], ['Dorsal', 'Ventral']):
        mask = groups == group_val
        
        # Scatter plot for this group
        curr_colors = fnames[mask].map(fname_to_idx)
        scatter = ax.scatter(x[mask], y[mask], z[mask], 
                            c=curr_colors, 
                            cmap=file_cmap,
                            vmin=0, vmax=len(unique_fnames)-1,
                            marker=marker, 
                            s=250, alpha=0.6, edgecolors='k')

    # Add ID labels to the center of every point
    for i in range(len(bbs2_all)):
        ax.text(x.iloc[i], y.iloc[i], z.iloc[i], 
                str(ids.iloc[i]),
                color='black', fontsize=8, fontweight='bold',
                ha='center', va='center')

    # Create the Legends
    # Legend A: Region (Shape)
    region_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                markeredgecolor='k', markersize=10, label='Dorsal'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                markeredgecolor='k', markersize=10, label='Ventral')
    ]

    # Legend B: File Names (Color)
    file_handles = []
    for name in unique_fnames:
        color = file_cmap(fname_to_idx[name] / max(1, len(unique_fnames)-1))
        handle = plt.Line2D([0], [0], marker='s', color='w', 
                            markerfacecolor=color, markersize=10, label=name)
        file_handles.append(handle)

    # Add Region Legend
    first_legend = ax.legend(handles=region_handles, title="Region", 
                            loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.add_artist(first_legend)

    ax.legend(handles=file_handles, title="File Names", 
            loc='upper left', bbox_to_anchor=(1.05, 0.85), ncol=1)

    # Final Styling
    ax.set_title(f"{MUTANT_NAME} Distribution", fontsize=20)
    ax.set_xlabel("Entropy", fontsize=16, labelpad=10)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=16, labelpad=10)
    ax.set_zlabel("Frequency (Hz)", fontsize=16, labelpad=10)

    plt.tight_layout()




# Quick utility to force close all plots
plt.show(block=False)
plt.pause(0.001)
input("hit [enter] to end.")
plt.close("all")
