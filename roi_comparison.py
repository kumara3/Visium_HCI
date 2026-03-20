import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x, y):
    """Effect size: Cohen's d"""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std


def mean_diff_ci(x, y, confidence=0.95):
    """Confidence interval for difference in means"""
    diff = np.mean(x) - np.mean(y)
    
    se = np.sqrt(
        np.var(x, ddof=1)/len(x) + np.var(y, ddof=1)/len(y)
    )
    
    df = len(x) + len(y) - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    
    margin = t_crit * se
    
    return diff, diff - margin, diff + margin


def compare_groups(df, group_col, value_col):
    """Compare all ROI pairs for one marker"""
    
    groups = df[group_col].unique()
    results = []
    
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            
            g1, g2 = groups[i], groups[j]
            
            x = df[df[group_col] == g1][value_col].dropna()
            y = df[df[group_col] == g2][value_col].dropna()
            
            if len(x) < 2 or len(y) < 2:
                continue
            
            # t-test
            t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
            
            # effect size
            d = cohens_d(x, y)
            
            # CI
            mean_diff, ci_low, ci_high = mean_diff_ci(x, y)
            
            results.append({
                "marker": value_col,
                "group1": g1,
                "group2": g2,
                "mean_diff": mean_diff,
                "effect_size_d": d,
                "p_value": p_val,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
    
    return pd.DataFrame(results)


def run_all_markers(df, group_col, markers):
    """Run comparisons for multiple markers"""
    
    all_results = []
    
    for m in markers:
        res = compare_groups(df, group_col, m)
        all_results.append(res)
    
    return pd.concat(all_results, ignore_index=True)