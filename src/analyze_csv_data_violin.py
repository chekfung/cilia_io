import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

'''
This file compiles all CSV files from different confocal microscope videos, 
and then starts doing statistical analysis between dorsal, ventral in WT and 
mutant zebrafish. It then generates violin plots, Welch T-Tests between groups,
and Cohen's power analysis.

NOTE: DORSAL = Group 0, Ventral = Group 1 :)
'''

# === Data folder ===
folder = '../data/test_cilia_io/'

SAVE_FIGURES = False
alpha = 0.05
ALPHA = alpha
power = 0.8
DORSAL_COLOR = (179, 110, 59)
VENTRAL_COLOR = (132, 172, 195)
run_arl_accumulation = True

# -----------

power_analysis = TTestIndPower()
bbs2_dfs = []
wt_dfs = []

def cv2_to_rgb(bgr):
    return tuple([c/255.0 for c in bgr[::-1]])

PALETTE = [cv2_to_rgb(DORSAL_COLOR), cv2_to_rgb(VENTRAL_COLOR)]

if SAVE_FIGURES:
    # Check if directory exists
    last_dir = os.path.basename(os.path.normpath(folder))
    figure_directory = os.path.join("../writing", last_dir)
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

# === Utils ===
def make_unique_cilia_id(df, prefix):
    df = df.copy()
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

# === Plotting ===
def plot_violin_and_ttest(df, feature, name, fname="", feat_unit=None):
    g0 = df[df['group'] == 0][feature].dropna()
    g1 = df[df['group'] == 1][feature].dropna()
    if g0.empty or g1.empty:
        print(f"Warning: One of the groups for {feature} in {fname} empty, skipping")
        return

    plt.figure(figsize=(3.5, 4))
    ax = sns.violinplot(
        x="group", y=feature, data=df,hue='group',
        palette=PALETTE, inner="box", cut=0,legend=False
    )
    plt.xticks([0, 1], ["Dorsal", "Ventral"])
    plt.title(f"{feature.replace('_',' ').title()} ({name})")
    plt.xlabel("")
    
    if feat_unit != None:
        plt.ylabel(feature.replace("_", " ").title() +" "+ feat_unit)
    else:
        plt.ylabel(feature.replace("_", " ").title())

    plt.tight_layout()
    sns.despine()

    # T-test
    t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
    stars = significance_stars(p_val)

    # Power Analysis
    n1, n2 = len(g0), len(g1)
    mean_diff = np.mean(g0) - np.mean(g1)
    s1, s2 = np.std(g0, ddof=1), np.std(g1, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    cohen_d = mean_diff/pooled_std
    ratio = n2/n1
    required_n = power_analysis.solve_power(effect_size=abs(cohen_d),
                                            alpha=alpha, power=power, ratio=ratio)

    print(f"{fname}, {feature} (Required N/GROUP: {required_n:.1f}, n={n1},{n2}) "
          f"T={t_stat:.4f}, p={p_val:.4e}, stars={stars}")

    # Determine the Number of Stars to Put on the Plot
    y_max = max(df[feature].dropna())
    y_min = min(df[feature].dropna())
    y, h = y_max + 0.1*(y_max-y_min), 0.05*(y_max-y_min)
    ax.plot([0,0,1,1], [y,y+h,y+h,y], lw=1.5, c='k')

    if stars != "ns":
        ax.text(0.5, y+h, stars, ha='center', va='bottom', fontsize=12)

    if SAVE_FIGURES:
        filename = os.path.join(figure_directory, f"{fname}_{feature}_dorsal_ventral_comparison.png")
        plt.savefig(filename, dpi=400)

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
            print(f"Warning: WT {feature}, group {group_map[g]} empty, skipping")
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
            comparisons.append(("accum_vs_wt", g_accum, g_wt))
        if not g_notaccum.empty:
            comparisons.append(("notaccum_vs_wt", g_notaccum, g_wt))
        if not g_accum.empty and not g_notaccum.empty:
            comparisons.append(("accum_vs_not_accum", g_accum, g_notaccum))

        y_max = data["value"].max()
        y_min = data["value"].min()
        y, h = y_max + 0.1*(y_max-y_min), 0.07*(y_max-y_min)

        for i, (label, mutant_vals, wt_vals) in enumerate(comparisons):
            t_stat, p_val = stats.ttest_ind(mutant_vals, wt_vals, equal_var=False)
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
                  f"T={t_stat:.4f}, p={p_val:.4e}, stars={stars}")

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
                "t": t_stat, "p": p_val, "cohen_d": cohen_d,
                "required_n": required_n, "n1": n1, "n2": n2
            }
            
        if SAVE_FIGURES:
            filename = os.path.join(figure_directory, f"arl_accum_compare_{feature}_{group_map[g]}_comparison.png")
            plt.savefig(filename, dpi=400)

        results[g] = group_results

    return results

def plot_violin_and_ttest_comparison_by_group(df0, df1, name0, name1, feature, group_col="group", feat_unit = None):
    group_map = {0: "Dorsal", 1: "Ventral"}  # map numbers to strings
    groups = sorted(pd.concat([df0[group_col], df1[group_col]]).dropna().unique())
    results = {}

    for g in groups:
        g0 = df0[df0[group_col]==g][feature].dropna()
        g1 = df1[df1[group_col]==g][feature].dropna()

        if g0.empty or g1.empty:
            print(f"Warning: {feature}, group {group_map[g]} empty in comparison {name0} vs {name1}, skipping")
            continue

        # Combine data for plotting
        data = pd.concat([
            pd.DataFrame({"value": g0, "dataset": name0}),
            pd.DataFrame({"value": g1, "dataset": name1})
        ], ignore_index=True)

        # Plot violin
        if g == 0:
            new_palette = [cv2_to_rgb(DORSAL_COLOR), cv2_to_rgb(DORSAL_COLOR)]
        else:
            new_palette = [cv2_to_rgb(VENTRAL_COLOR), cv2_to_rgb(VENTRAL_COLOR)]

        plt.figure(figsize=(3.5, 4))
        ax = sns.violinplot(
            x="dataset", y="value", data=data, hue="dataset",
            palette=new_palette, inner="box", cut=0, legend=False
        )

        # Define hatch patterns
        hatch_patterns = ["///", "..."]  

        # Go through each violin collection and apply hatches
        for i, collection in enumerate(ax.collections):
            if i==0:
                continue
            hatch = hatch_patterns[(i // 2) % len(hatch_patterns)]
            collection.set_hatch(hatch)
            collection.set_edgecolor("black")   # make hatch visible

        plt.title(f"{feature.replace('_',' ').title()} ({group_map[g]})")
        plt.xlabel("")

        if feat_unit != None:
            plt.ylabel(feature.replace("_", " ").title() + " "+feat_unit)
        else:
            plt.ylabel(feature.replace("_", " ").title())

        plt.tight_layout()
        sns.despine()

        # T-test
        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
        stars = significance_stars(p_val)

        # Power Analysis
        n1, n2 = len(g0), len(g1)
        mean_diff = g0.mean() - g1.mean()
        s1, s2 = g0.std(ddof=1), g1.std(ddof=1)
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        cohen_d = mean_diff / pooled_std
        ratio = n2 / n1
        required_n = power_analysis.solve_power(
            effect_size=abs(cohen_d), alpha=alpha, power=power, ratio=ratio
        )

        print(f"{name0} vs {name1}, {feature}, {group_map[g]} "
              f"(Required N/group: {required_n:.1f}, n={n1},{n2}) "
              f"T={t_stat:.4f}, p={p_val:.4e}, stars={stars}")

        # Significance stars
        y_max, y_min = data["value"].max(), data["value"].min()
        y, h = y_max + 0.1*(y_max-y_min), 0.05*(y_max-y_min)
    
        if stars != "ns":
            ax.text(0.5, y+h, stars, ha="center", va="bottom", fontsize=12)
            ax.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c="k")

        if SAVE_FIGURES:
            filename = os.path.join(figure_directory, f"compare_bbs2_wt_{feature}_{group_map[g]}.png")
            plt.savefig(filename, dpi=400)

        results[g] = {"t": t_stat, "p": p_val, "cohen_d": cohen_d, "required_n": required_n}

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
        if "bbs2" in fname.lower():
            bbs2_dfs.append(df); num_bbs2+=1
        elif "wt" in fname.lower():
            wt_dfs.append(df); num_wt+=1

# Concatenate all the different CSV files together
bbs2_all = pd.concat(bbs2_dfs, ignore_index=True) if bbs2_dfs else pd.DataFrame()
wt_all = pd.concat(wt_dfs, ignore_index=True) if wt_dfs else pd.DataFrame()

# Add a column called "beating_regularity, which is 1 - entropy"
if not bbs2_all.empty:
    bbs2_all["beating_regularity"] = 1 - bbs2_all["entropy"]
if not wt_all.empty:
    wt_all["beating_regularity"] = 1 - wt_all["entropy"]

print(f"Loaded {len(bbs2_all)} rows from bbs2 files")
print(f"Loaded {len(wt_all)} rows from wt files")

try:    
    print(f"BBS2 ARL Accumulation: {bbs2_all['arl13b_accumulation'].sum()}, No Accumulation: {len(bbs2_all) - bbs2_all['arl13b_accumulation'].sum()}")
except:
    print("No ARL Accumulation Data")
    run_arl_accumulation = False

# Save filtered CSVs
if not bbs2_all.empty:
    bbs2_all.to_csv(os.path.join(folder,"1_bbs2_all_filtered.csv"), index=False)
if not wt_all.empty:
    wt_all.to_csv(os.path.join(folder,"1_wt_all_filtered.csv"), index=False)

feat_list = ["frequency", "centroid_frequency","mean_amplitude","max_amplitude",
             "entropy","beating_regularity", "eccentricity","average_speed","mean_cilia_length","mean_cilia_area",
             "mean_straightness_flipped","std_straightness_flipped","max_straightness_flipped","min_straightness_flipped"]

feat_unit = ["(Hz)", "(Hz)", "($\mu$m)", "($\mu$m)", None, None, None, "($\mu$m/sec)", "($\mu$m)", "($\mu$m$^2$)",
             None, None, None, None]

# === Analyses ===
if not bbs2_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_ttest(bbs2_all, feat, "bbs2", fname="bbs2", feat_unit=feat_unit[i])

if not wt_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_ttest(wt_all, feat, "wt", fname="wt", feat_unit=feat_unit[i])

if not bbs2_all.empty and not wt_all.empty:
    for i, feat in enumerate(feat_list):
        plot_violin_and_ttest_comparison_by_group(wt_all, bbs2_all, "wt", "bbs2", feat, feat_unit=feat_unit[i])

    if run_arl_accumulation:
        for i, feat in enumerate(feat_list):
            plot_violin_arl_accumulation_two_groups(wt_all, bbs2_all, "wt", "bbs2", feat, feat_unit[i])

# === Keep your PCA / clustering code intact ===
if not wt_all.empty:
    count_0 = (wt_all['group'] == 0).sum()
    count_1 = (wt_all['group'] == 1).sum()

    fig = plt.figure()
    colormap_str = "viridis"
    DORSAL_COLOR = (179, 110, 59)
    VENTRAL_COLOR = (132, 172, 195)

    PALETTE = [cv2_to_rgb(DORSAL_COLOR), cv2_to_rgb(VENTRAL_COLOR)]

    x = wt_all['beating_regularity']
    y = wt_all["mean_amplitude"]
    z = wt_all["frequency"]
    groups = wt_all['group']

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=groups, cmap=mcolors.ListedColormap(PALETTE), s=200, alpha=0.6,edgecolors='k')
    ax.set_title(f"WT Groups (Dorsal Vs. Ventral) (N={num_wt}) Dorsal: {count_0} Ventral: {count_1}", fontsize=20)

    ax.set_xlabel("Beating Regularity", fontsize=18, labelpad=15)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=18, labelpad=15)
    ax.set_zlabel("Frequency (Hz)", fontsize=18, labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # Manually assign legend using dummy points
    cmap = mcolors.ListedColormap(PALETTE)
    group_to_label = {0: "Dorsal", 1: "Ventral"}

    for group_val, label in group_to_label.items():
        ax.scatter([], [], [],  # Invisible point
                color=cmap(group_val / 1),  # 0.0 or 1.0
                label=label,
                s=60, alpha=0.6)

    ax.legend(fontsize=14, markerscale=1.5)

    # Combine the 3 lists into a 3D dataset
    data_3d = np.array([x, y, z]).T

    # Column-wise Normalization
    scaler = StandardScaler()
    data_3d = scaler.fit_transform(data_3d)

    # Perform PCA to reduce to 1 dimension
    pca = PCA(n_components=1)  # Initialize PCA with 1 component
    pca.fit(data_3d)           # Fit PCA to your 3D data
    data_1d = pca.transform(data_3d) # Transform the 3D data to 1D
    print("beating_regularity , max_amplitude , frequency")
    print(pca.components_)

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

    # t-test for PC1 between groups ---
    pc1_dorsal = data_1d[groups == 0].ravel()
    pc1_ventral = data_1d[groups == 1].ravel()
    t_stat, p_val = stats.ttest_ind(pc1_dorsal, pc1_ventral, equal_var=False)

    print(f"T-test: t = {t_stat:.6f}, p = {p_val:.8e}")

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
    count_0 = (bbs2_all['group'] == 0).sum()
    count_1 = (bbs2_all['group'] == 1).sum()

    fig = plt.figure()
    colormap_str = "viridis"
    DORSAL_COLOR = (179, 110, 59)
    VENTRAL_COLOR = (132, 172, 195)

    PALETTE = [cv2_to_rgb(DORSAL_COLOR), cv2_to_rgb(VENTRAL_COLOR)]

    x = bbs2_all['beating_regularity']
    y = bbs2_all["mean_amplitude"]
    z = bbs2_all["frequency"]
    groups = bbs2_all['group']

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=groups, cmap=mcolors.ListedColormap(PALETTE), s=200, alpha=0.6,edgecolors='k')
    ax.set_title(f"BBS2 Groups (Dorsal Vs. Ventral) (N={num_bbs2}) Dorsal: {count_0} Ventral: {count_1}", fontsize=20)

    ax.set_xlabel("Beating Regularity", fontsize=18, labelpad=15)
    ax.set_ylabel("Mean Amplitude ($\mu$m)", fontsize=18, labelpad=15)
    ax.set_zlabel("Frequency (Hz)", fontsize=18, labelpad=15)

    ax.tick_params(axis='both', which='major', labelsize=18)

    # Manually assign legend using dummy points
    cmap = mcolors.ListedColormap(PALETTE)
    group_to_label = {0: "Dorsal", 1: "Ventral"}

    for group_val, label in group_to_label.items():
        ax.scatter([], [], [],  # Invisible point
                color=cmap(group_val / 1),  # 0.0 or 1.0
                label=label,
                s=60, alpha=0.6)

    ax.legend(fontsize=14, markerscale=1.5)

    # Combine the 3 lists into a 3D dataset
    data_3d = np.array([x, y, z]).T

    # Column-wise Normalization
    scaler = StandardScaler()
    data_3d = scaler.fit_transform(data_3d)

    # Perform PCA to reduce to 1 dimension
    pca = PCA(n_components=1)  # Initialize PCA with 1 component
    pca.fit(data_3d)           # Fit PCA to your 3D data
    data_1d = pca.transform(data_3d) # Transform the 3D data to 1D
    print("beating_regularity , max_amplitude , frequency")
    print(pca.components_)

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

    plt.title('PCA Dimension 1 Histogram by Group (bbs2)', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Number of Cilia', fontsize=14)
    plt.legend(['Dorsal', 'Ventral'], fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Logistic regression with statsmodels
    X = sm.add_constant(data_1d)  # add intercept
    logit_model = sm.Logit(groups, X)
    result = logit_model.fit()

    print("BBS2 Logistic Regression PCA")
    print(result.summary())

    # --- t-test for PC1 between groups ---
    pc1_dorsal = data_1d[groups == 0].ravel()
    pc1_ventral = data_1d[groups == 1].ravel()
    t_stat, p_val = stats.ttest_ind(pc1_dorsal, pc1_ventral, equal_var=False)

    print(f"BBS2 PCA T-test: t = {t_stat:.6f}, p = {p_val:.8e}")

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

plt.show(block=False)
plt.pause(0.001)
input("hit [enter] to end.")
plt.close("all")
