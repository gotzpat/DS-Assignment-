# --- IMPORTS ---
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, chi2_contingency
import os, re

# ---------- CONFIG: PLOTS FOLDER ----------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def safe_name(s: str) -> str:
    """Make a string safe for filenames."""
    s = re.sub(r"[^\w\s\-]+", "", str(s))
    s = re.sub(r"\s+", "_", s.strip())
    return s[:120]

def savefig_in_plots(filename: str, dpi=300):
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

def detect_outliers_intelligent(df, var_type_dict):
    """
    Detect outliers based on variable type.
    
    Parameters:
    -----------
    df : DataFrame
    var_type_dict : dict mapping variable types to column lists
    
    Returns:
    --------
    DataFrame with outlier summary
    """
    results = []
    
    # 1. Skip binary variables entirely
    print("\n📊 Binary variables (skipping outlier detection):")
    for col in var_type_dict.get('binary', []):
        if col in df.columns:
            print(f"  - {col}: values = {sorted(df[col].dropna().unique())}")
    
    # 2. Ordinal categorical: check for unexpected values
    print("\n📊 Ordinal/Categorical variables (checking for invalid codes):")
    for col in var_type_dict.get('ordinal', []):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        unique_vals = sorted(series.unique())
        print(f"  - {col}: {len(unique_vals)} categories: {unique_vals}")
    
    # 3. Grade variables: check for values outside valid range
    print("\n📊 Grade variables (checking range violations):")
    for col in var_type_dict.get('grades', []):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        # Assuming grades are 0-200 scale (common in some education systems)
        min_val, max_val = series.min(), series.max()
        invalid = ((series < 0) | (series > 200)).sum()
        invalid_pct = 100 * invalid / len(series) if len(series) > 0 else 0
        
        print(f"  - {col}: range=[{min_val:.1f}, {max_val:.1f}], "
              f"invalid (outside 0-20): {invalid} ({invalid_pct:.1f}%)")
        
        if invalid > 0:
            results.append({
                'column': col,
                'type': 'grade',
                'issue': 'out_of_range',
                'count': invalid,
                'pct': invalid_pct
            })
    
    # 4. Count variables: use more lenient threshold (e.g., 3*IQR or z-score > 4)
    print("\n📊 Count variables (using lenient outlier detection):")
    for col in var_type_dict.get('counts', []):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        # More lenient: 3*IQR instead of 1.5*IQR
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        outlier_pct = 100 * outliers / len(series)
        
        print(f"  - {col}: Q1={Q1:.1f}, Q3={Q3:.1f}, "
              f"extreme outliers (3*IQR): {outliers} ({outlier_pct:.1f}%)")
        
        if outliers > 0:
            results.append({
                'column': col,
                'type': 'count',
                'issue': 'extreme_outlier_3IQR',
                'count': outliers,
                'pct': outlier_pct
            })
    
    # 5. Continuous variables: standard IQR method
    print("\n📊 Continuous variables (standard IQR outlier detection):")
    for col in var_type_dict.get('continuous', []):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        outlier_pct = 100 * outliers / len(series)
        
        print(f"  - {col}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}, "
              f"outliers: {outliers} ({outlier_pct:.1f}%)")
        
        if outliers > 0:
            results.append({
                'column': col,
                'type': 'continuous',
                'issue': 'outlier_1.5IQR',
                'count': outliers,
                'pct': outlier_pct
            })
    
    return pd.DataFrame(results)

def clean_dataframe(df, col_missing_thresh=0.30, row_missing_thresh=0.50):
    """Basic cleaning:
    - report missing values
    - drop columns with > col_missing_thresh missing fraction
    - drop rows with > row_missing_thresh missing fraction
    - try to coerce numeric types
    - map common yes/no-like strings to 0/1
    - impute numeric with median and categorical with mode
    """
    df = df.copy()
    print("\nMissing values (before):")
    missing = df.isna().sum()
    print(missing[missing > 0])

    # drop columns with too many missing values
    col_frac = df.isna().mean()
    drop_cols = col_frac[col_frac > col_missing_thresh].index.tolist()
    if drop_cols:
        print(f"Dropping columns with >{int(col_missing_thresh*100)}% missing:", drop_cols)
        df.drop(columns=drop_cols, inplace=True)

    # drop rows with too many missing values
    row_frac = df.isna().mean(axis=1)
    drop_rows = row_frac[row_frac > row_missing_thresh].index
    if len(drop_rows):
        print(f"Dropping {len(drop_rows)} rows with >{int(row_missing_thresh*100)}% missing")
        df = df.drop(index=drop_rows).reset_index(drop=True)

    # try to coerce numeric where appropriate
    df = df.apply(lambda s: pd.to_numeric(s, errors="ignore"))

    # simple imputation: numeric -> median, categorical -> mode
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    for col in df.select_dtypes(include=["category","object"]).columns:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    print("\nMissing values (after):")
    missing_after = df.isna().sum()
    print(missing_after[missing_after > 0] if missing_after.any() else "None")
    print("Data shape after cleaning:", df.shape)
    return df

# ---------- YOUR SELECTED COLUMNS ----------
selected_columns = [ 
    "Marital Status", 
    "Application order", 
    "Course", 
    "Daytime/evening attendance", 
    "Previous qualification", 
    "Previous qualification (grade)",
    "Nacionality", 
    "Mother's qualification", 
    "Father's qualification", 
    "Mother's occupation", 
    "Father's occupation", 
    "Admission grade", 
    "Educational special needs", 
    "Gender", 
    "Scholarship holder", 
    "Age at enrollment", 
    "Displaced", 
    "Debtor", 
    "International", 
    "Curricular units 1st sem (credited)", 
    #"Curricular units 1st sem (enrolled)", # commented out based on corrlation analysis
    #"Curricular units 1st sem (evaluations)",
    #"Curricular units 1st sem (approved)", 
    "Curricular units 1st sem (grade)", 
    #"Curricular units 1st sem (without evaluations)", 
    "Curricular units 2nd sem (credited)", 
    #"Curricular units 2nd sem (enrolled)", 
    #"Curricular units 2nd sem (evaluations)", 
    #"Curricular units 2nd sem (approved)", 
    "Curricular units 2nd sem (grade)", 
    #"Curricular units 2nd sem (without evaluations)", 
    "Unemployment rate", 
    "Inflation rate", 
    "GDP", 
    "Target", 
]

# ---------- CORRECTED VARIABLE TYPE GROUPS ----------
# Binary variables (0/1)
binary_vars = [
    "Daytime/evening attendance",  # 1=day, 0=evening
    "Educational special needs", 
    "Gender", 
    "Scholarship holder",
    "Displaced", 
    "Debtor", 
    "International"
]

# Nominal categorical (no natural ordering)
nominal_categorical = [
    "Course",         # Different programs (Nursing, Engineering, etc.)
    "Nacionality",    # Different countries
]

# Ordinal categorical (meaningful order/hierarchy)
ordinal_categorical = [
    "Marital Status",           # Some ordering possible
    "Application order",        # 0 (first choice) to 9 (last choice)
    "Previous qualification",   # Education levels (ordered)
    "Mother's qualification",   # Education levels (ordered)
    "Father's qualification",   # Education levels (ordered)
    "Mother's occupation",      # Likely some hierarchy
    "Father's occupation",      # Likely some hierarchy
]

# Grade/score variables (0-200 scale per UCI docs!)
grade_vars = [
    "Previous qualification (grade)", 
    "Admission grade",
    "Curricular units 1st sem (grade)", 
    "Curricular units 2nd sem (grade)"
]

# Count variables (non-negative integers, right-skewed)
count_vars = [
    #"Curricular units 1st sem (credited)", 
    "Curricular units 1st sem (enrolled)",
    #"Curricular units 1st sem (evaluations)", 
    #"Curricular units 1st sem (approved)",
    #"Curricular units 1st sem (without evaluations)",
    #"Curricular units 2nd sem (credited)", 
    "Curricular units 2nd sem (enrolled)",
    #"Curricular units 2nd sem (evaluations)", 
    #"Curricular units 2nd sem (approved)",
    #"Curricular units 2nd sem (without evaluations)"
]

# Continuous variables (real-valued)
continuous_vars = [
    "Age at enrollment",    # Age in years
    "Unemployment rate",    # Percentage
    "Inflation rate",       # Percentage
    "GDP"                   # Economic indicator
]


# --- 1. LOAD DATA ---
dataset = fetch_ucirepo(id=697)
X = np.array(dataset.data.features)
y = np.array(dataset.data.targets)

# --- 2. CREATE DATAFRAME ---
col_names = dataset.variables["name"]
merged_df = pd.DataFrame(np.column_stack((X, y)), columns=col_names)

# --- 3. CLEAN DATA ---
print("\n===== Cleaning Data =====")
print("Data shape:", merged_df.shape)
print("Missing values per column:\n", merged_df.isnull().sum())
# select only relevant columns
merged_df = merged_df[selected_columns].copy()
# call the cleaner right after creating merged_df
merged_df = clean_dataframe(merged_df, col_missing_thresh=0.30, row_missing_thresh=0.50)

# --- 4. MISSING VALUES SUMMARY ---
print("\n===== Missing Values Summary =====")
missing_summary = (
    merged_df.isnull().sum()
    .to_frame("missing_count")
    .assign(
        total_rows=len(merged_df),
        missing_pct=lambda df: 100 * df["missing_count"] / df["total_rows"]
    )
    .query("missing_count > 0")
    .sort_values("missing_pct", ascending=False)
)
if missing_summary.empty:
    print("✅ No missing values remaining.")
else:
    print(missing_summary)
    missing_summary_path = os.path.join(PLOTS_DIR, "missing_values_summary.csv")
    missing_summary.to_csv(missing_summary_path)
    print(f"[saved] {missing_summary_path}")

# --- 5. OUTLIER DETECTION (IQR METHOD) ---
print("\n===== Outlier Detection (Numeric Columns) =====")
var_types = {
    'binary': binary_vars,
    'nominal': nominal_categorical,
    'ordinal': ordinal_categorical,
    'grades': grade_vars,
    'counts': count_vars,
    'continuous': continuous_vars
}

# Run intelligent outlier detection
outlier_results = detect_outliers_intelligent(merged_df, var_types)

# Save results
if not outlier_results.empty:
    outlier_results = outlier_results.sort_values('pct', ascending=False)
    print("\n📋 Summary of detected issues:")
    print(outlier_results)
    
    outlier_path = os.path.join(PLOTS_DIR, "intelligent_outlier_summary.csv")
    outlier_results.to_csv(outlier_path, index=False)
    print(f"[saved] {outlier_path}")
    
    # Visualize only the problematic variables
    for _, row in outlier_results.iterrows():
        col = row['column']
        plt.figure(figsize=(10, 4))
        
        # Create subplot with histogram and boxplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        merged_df[col].hist(bins=30, ax=ax1)
        ax1.set_title(f"{col} - Distribution")
        ax1.set_xlabel(col)
        ax1.set_ylabel("Frequency")
        
        # Boxplot
        sns.boxplot(y=merged_df[col], ax=ax2)
        ax2.set_title(f"{col} - Boxplot ({row['type']})")
        
        plt.suptitle(f"{col}: {row['count']} potential issues ({row['pct']:.1f}%)")
        savefig_in_plots(f"outliers_intelligent_{safe_name(col)}.png")
else:
    print("\n✅ No significant outliers detected with intelligent method!")

# --- 6. CHECK TARGET VARIABLE ---
print("\n===== TARGET VALUE =====")
target_col = "Target"
print("\nTarget distribution (raw codes):")
print(merged_df[target_col].value_counts(dropna=False))

# Optional: rename target categories for readability (keep as string/categorical)
merged_df[target_col] = merged_df[target_col].replace({
    0: "Dropout", 
    1: "Enrolled", 
    2: "Graduate"
})
# Ensure it's treated as a categorical with a sensible order for plots
merged_df[target_col] = pd.Categorical(
    merged_df[target_col], 
    categories=["Dropout", "Enrolled", "Graduate"], 
    ordered=True
)

# --- 7. DESCRIPTIVE STATS ---
print("\n===== DESCRIPTIVE STATISTICS =====")
print("\nDescriptive statistics for numeric columns:")
numeric_cols = merged_df.select_dtypes(include=np.number).columns
desc = merged_df[numeric_cols].describe().T
print(desc)
# Save descriptive stats
desc_path = os.path.join(PLOTS_DIR, "descriptive_stats_numeric.csv")
desc.to_csv(desc_path, index=True)
print(f"[saved] {desc_path}")

# --- 8. CORRELATION ANALYSIS ---
print("\n===== CORRELATION ANALYSIS =====")
plt.figure(figsize=(14, 12))
corr = merged_df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Numeric Variables", fontsize=14)
savefig_in_plots("correlation_matrix.png")

# Also save the correlation matrix as CSV
corr_path = os.path.join(PLOTS_DIR, "correlation_matrix.csv")
corr.to_csv(corr_path)
print(f"[saved] {corr_path}")

# # --- 9. FEATURE IMPORTANCE (BASIC INSIGHTS via ANOVA) ---
print("\n===== CFEATURE IMPORTANCE =====")
anova_results = {}
numeric_cols = merged_df.select_dtypes(include=np.number).columns  # target is categorical now
print("Numeric columns considered for ANOVA:", len(numeric_cols))

for col in numeric_cols:
    groups = [merged_df.loc[merged_df[target_col] == cat, col].dropna()
              for cat in merged_df[target_col].cat.categories
              if cat in merged_df[target_col].unique()]
    # need at least 2 non-empty groups
    if sum(len(g) > 0 for g in groups) < 2:
        continue

    f_val, p_val = f_oneway(*groups)

    # effect size: eta-squared = SS_between / SS_total
    grand_mean = merged_df[col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((merged_df[col] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    anova_results[col] = {"p_value": p_val, "eta_sq": eta_sq}

anova_df = (pd.DataFrame(anova_results).T
            .sort_values(["p_value", "eta_sq"], ascending=[True, False]))
anova_df["significant"] = anova_df["p_value"] < 0.05
print("\nVariables most significantly different between target groups:")
print(anova_df.head(10))

anova_path = os.path.join(PLOTS_DIR, "anova_by_target.csv")
anova_df.to_csv(anova_path, index=True)
print(f"[saved] {anova_path}")

# --- 10. VISUALIZATIONS FOR IMPORTANT VARIABLES ---
print("\n===== VISUALIZATIONS FOR IMPORTANT VARIABLES =====")
important_vars = anova_df.loc[anova_df["significant"]].head(10).index.tolist()

for var in important_vars:
    # Determine if the variable is binary (two unique non-NA values)
    non_na = merged_df[var].dropna()
    is_binary = non_na.nunique() == 2

    if is_binary:
        # Stacked bar of proportions within each target category
        plt.figure(figsize=(7, 5))
        # Build normalized (row-wise) contingency table: Target x var
        tab = (pd.crosstab(merged_df[target_col], merged_df[var])
                 .apply(lambda r: r / r.sum(), axis=1))
        # Ensure columns ordered as 0,1 if they exist
        tab = tab.reindex(columns=sorted(tab.columns.tolist()))
        tab.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.ylabel("Proportion within target group")
        plt.title(f"{var} vs {target_col} (stacked proportions)")
        plt.legend(title=var, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.ylim(0, 1)
        savefig_in_plots(f"stacked_binary_{safe_name(var)}by{safe_name(target_col)}.png")
    else:
        # Boxplot for continuous/multi-level numeric variables
        plt.figure(figsize=(7, 5))
        sns.boxplot(x=target_col, y=var, data=merged_df)
        plt.title(f"{var} by {target_col}")
        savefig_in_plots(f"boxplot_{safe_name(var)}.png")


# --- 11. CATEGORICAL VARIABLES RELATIONSHIP (Chi-square) ---
print("\n===== CATEGORICAL VARIABLES RELATIONSHIP (Chi-square) =====")
categorical_cols = merged_df.select_dtypes(exclude=[np.number]).columns.drop(target_col, errors="ignore")

chi2_rows = []
for col in categorical_cols:
    tab = pd.crosstab(merged_df[col], merged_df[target_col])
    # need at least 2x2 contingency for chi-square
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        continue
    chi2, p, dof, exp = chi2_contingency(tab)
    chi2_rows.append((col, p, dof, tab.sum().sum()))

chi2_df = pd.DataFrame(chi2_rows, columns=["variable","p_value","dof","n"]).sort_values("p_value")
print("\nCategorical variables significantly related to target:")
print(chi2_df[chi2_df["p_value"] < 0.05])

# Save chi-square table
chi2_path = os.path.join(PLOTS_DIR, "chi2_by_target.csv")
chi2_df.to_csv(chi2_path, index=False)
print(f"[saved] {chi2_path}")

# barplots for top-3 categorical variables (lowest p-values)
for var in chi2_df.head(3)["variable"]:
    plt.figure(figsize=(7, 5))
    (merged_df.groupby([var, target_col])
              .size()
              .groupby(level=0)
              .apply(lambda s: s/s.sum())
              .unstack()
              .plot(kind="bar", stacked=True, ax=plt.gca()))
    plt.ylabel("Share within category")
    plt.title(f"{var} vs {target_col} (stacked share)")
    plt.legend(title=target_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig_in_plots(f"stacked_bar_{safe_name(var)}by{safe_name(target_col)}.png")

# --- 13. TARGET DISTRIBUTION PLOT ---
plt.figure(figsize=(6, 4))
sns.countplot(x=target_col, data=merged_df)
plt.title("Target Variable Distribution")
savefig_in_plots("target_distribution.png")

print("\nAll figures and tables saved under:", os.path.abspath(PLOTS_DIR))


