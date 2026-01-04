import pandas as pd
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION (The "Control Panel")
# ==========================================
# Researchers can change these settings without touching the logic

CONFIG = {
    # Paths
    'hh_data': "data/23-24/hhv1.dta",
    'per_data': "data/23-24/perv1.dta",
    'layout': "manuals/Data_LayoutPLFS_2023-24.xlsx",
    'hh_sheet': "hhv1",
    'per_sheet': "perv1",
    'output_file': "processed_urban_women_2324.csv",

    # Merge Keys (Must match exactly)
    'keys': ['mfsu', 'sec', 'ssu', 'visit', 'seg', 'sss'],

    # Columns to Keep (The "Menu")
    'hh_cols': ['st', 'dc', 'relg', 'sg', 'hh_size', 'hhtype', 'hce_tot', 'mult'],
    'per_cols': [
        'rel', 'sex', 'age', 'marst',       # Demographics
        'gedu_lvl', 'tedu_lvl', 'form_edu', # Education
        'pas', 'ind_pas', 'ocu_pas',        # Work Codes
        'ern_reg', 'ern_self', 'mult'       # Wages & Weight
    ],

    # Target Population Filters
    'filters': {
        'sector': 2,      # 1=Rural, 2=Urban
        'sex': 2,         # 1=Male, 2=Female
        'marst': 2,       # 2=Married
        'age_min': 15,
        'age_max': 60
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_and_rename(data_path, layout_path, sheet_name):
    """Loads Stata file and renames columns based on Excel layout."""
    print(f"--- Loading: {os.path.basename(data_path)} ({sheet_name}) ---")
    
    # Load Data
    try:
        df = pd.read_stata(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found: {data_path}")

    # Load Layout
    layout = pd.read_excel(layout_path, sheet_name=sheet_name)
    target_col = 'Field_Name'
    
    if target_col not in layout.columns:
        raise ValueError(f"❌ Layout sheet missing '{target_col}' column.")
        
    # Create Mapping
    valid_names = layout[target_col].dropna().astype(str).str.strip().tolist()
    
    if len(df.columns) != len(valid_names):
        print(f"⚠️ WARNING: Length mismatch ({len(df.columns)} vs {len(valid_names)}). Check layout file.")
        # Proceeding strictly might be dangerous, but usually safe for standard MoSPI files
    
    rename_map = dict(zip(df.columns, valid_names))
    df = df.rename(columns=rename_map)
    print(f"✅ Loaded {len(df)} rows, Renamed {len(df.columns)} columns.")
    return df

def feature_engineering(df):
    """Calculates MPCE, Wages, and Labor Flags."""
    print("--- Running Feature Engineering ---")
    
    # 1. Weights (Scale down by 100)
    # We use 'mult_x' because it comes from the Person file in the merge
    if 'mult_x' in df.columns:
        df['weight'] = df['mult_x'] / 100
    
    # 2. Wealth (MPCE)
    df['mpce'] = df['hce_tot'] / df['hh_size']
    
    # 3. Wages (Regular + Self)
    df['total_wage'] = df['ern_reg'].fillna(0) + df['ern_self'].fillna(0)
    
    # 4. Labor Status Flags
    df['pas'] = pd.to_numeric(df['pas'], errors='coerce')
    df['is_employed'] = df['pas'].between(11, 51).astype(int)
    df['is_unemployed'] = df['pas'].between(81, 82).astype(int)
    df['is_domestic_duties'] = df['pas'].isin([92, 93]).astype(int)
    
    return df

def match_spouses(df_full, df_target, keys):
    """Finds husbands in the full dataset and attaches wage to target women."""
    print("--- Matching Spouses ---")
    
    # 1. Ensure numeric types for matching
    cols_to_fix = ['rel', 'sex', 'marst', 'total_wage']
    for c in cols_to_fix:
        df_full[c] = pd.to_numeric(df_full[c], errors='coerce')
        # Target might already be processed, but safety first
        if c in df_target.columns:
            df_target[c] = pd.to_numeric(df_target[c], errors='coerce')

    # 2. Identify Potential Husbands (Male, Married, Head/Spouse)
    df_men = df_full[
        (df_full['sex'] == 1) & 
        (df_full['marst'] == 2) & 
        (df_full['rel'].isin([1, 2]))
    ].copy()

    # 3. Handle Joint Families (Keep highest earner per household)
    spouse_cols = keys + ['total_wage']
    df_husbands = df_men[spouse_cols].rename(columns={'total_wage': 'spousal_wage'})
    df_husbands = df_husbands.sort_values('spousal_wage', ascending=False).drop_duplicates(subset=keys)

    # 4. Merge
    # Ensure keys are strings
    for k in keys:
        df_target[k] = df_target[k].astype(str)
        df_husbands[k] = df_husbands[k].astype(str)

    df_final = pd.merge(df_target, df_husbands, on=keys, how='left')
    df_final['spousal_wage'] = df_final['spousal_wage'].fillna(0)
    
    print(f"✅ Matched Spouses. Found matches for {len(df_husbands)} households.")
    return df_final

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def main():
    # A. Load
    df_hh = load_and_rename(CONFIG['hh_data'], CONFIG['layout'], CONFIG['hh_sheet'])
    df_per = load_and_rename(CONFIG['per_data'], CONFIG['layout'], CONFIG['per_sheet'])

    # B. Filter Columns (Keep only strict list)
    hh_keep = CONFIG['keys'] + CONFIG['hh_cols']
    per_keep = CONFIG['keys'] + CONFIG['per_cols']
    
    # Safe Select
    df_hh = df_hh[[c for c in hh_keep if c in df_hh.columns]]
    df_per = df_per[[c for c in per_keep if c in df_per.columns]]

    # C. Merge Master Dataset
    print("--- Merging Files ---")
    for k in CONFIG['keys']:
        df_hh[k] = df_hh[k].astype(str)
        df_per[k] = df_per[k].astype(str)

    df_master = pd.merge(df_per, df_hh, on=CONFIG['keys'], how='left', validate='m:1')
    
    # D. Feature Engineering (On Master)
    df_master = feature_engineering(df_master)

    # E. Filter Target Population
    print("--- Filtering Target Population ---")
    filters = CONFIG['filters']
    
    # Ensure numeric for filtering
    for col in ['sec', 'sex', 'marst', 'age']:
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    df_target = df_master[
        (df_master['sec'] == filters['sector']) &
        (df_master['sex'] == filters['sex']) &
        (df_master['marst'] == filters['marst']) &
        (df_master['age'].between(filters['age_min'], filters['age_max']))
    ].copy()
    
    # F. Match Spouses
    # Note: We pass df_master (to find men) and df_target (to attach them)
    df_final = match_spouses(df_master, df_target, CONFIG['keys'])

    # G. Cleanup & Save
    # Remove technical merge artifacts
    drop_cols = ['mult_x', 'mult_y']
    df_final.drop(columns=[c for c in drop_cols if c in df_final.columns], inplace=True)
    
    print(f"\n✅ PROCESSING COMPLETE.")
    print(f"Final Shape: {df_final.shape}")
    
    # Save
    df_final.to_csv(CONFIG['output_file'], index=False)
    print(f"Saved to: {CONFIG['output_file']}")

if __name__ == "__main__":
    main()