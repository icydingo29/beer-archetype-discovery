import numpy as np

def calculate_cluster_sigmas(X, U, centroids, m):
    # sigmas define the width of the Gaussian membership function.
    # they are calculated as weighted standard deviations of the points in each cluster.
    # X: (N, 7), U: (C, N), centroids: (C, 7)
    Um = U ** m
    sigmas = np.zeros_like(centroids)
    
    for i in range(centroids.shape[0]):
        diff_sq = (X - centroids[i]) ** 2 # (N, 7)
        
        # weighted sum of squares of memberships
        numerator = np.sum(Um[i][:, None] * diff_sq, axis=0) # (7,)
        denominator = np.sum(Um[i])
        
        sigmas[i] = np.sqrt(numerator / denominator)
    
    return sigmas

def get_beer_membership_adaptive(beer_idx, beer_profiles, centroids, sigmas):
    x = beer_profiles[beer_idx]

    diff_sq = (x - centroids) ** 2
    exponent = -diff_sq / (2 * (sigmas ** 2))
    
    # we use Log-Sum-Exp logic for numerical stability 
    # summing log-exponents is equivalent to multiplying the original membership values (Product T-norm)
    log_activations = np.sum(exponent, axis=1)
    
    # firing strengths
    exp_acts = np.exp(log_activations)
    total_activation = np.sum(exp_acts)
    
    if total_activation == 0:
        return np.ones(len(centroids)) / len(centroids)
        
    return exp_acts / total_activation

def generate_linguistic_rules_denorm(features, centroids, num_archetypes, original_df):
    print("\n=== GENERATED LINGUISTIC RULES (Denormalized & Granular) ===")
    
    feature_stats = {}
    for feat in features:
        feature_stats[feat] = {
            'min': original_df[feat].min(),
            'max': original_df[feat].max(),
            'q20': original_df[feat].quantile(0.20),
            'q40': original_df[feat].quantile(0.40),
            'q60': original_df[feat].quantile(0.60),
            'q80': original_df[feat].quantile(0.80)
        }

    for i in range(num_archetypes):
        rule_parts = []
        for j, feat in enumerate(features):
            # denormalize
            f_min = feature_stats[feat]['min']
            f_max = feature_stats[feat]['max']
            val_norm = centroids[i, j]
            val_denorm = val_norm * (f_max - f_min) + f_min
            
            # 5 categories
            stats = feature_stats[feat]
            if val_denorm <= stats['q20']:
                hedge = "Very Low"
            elif val_denorm <= stats['q40']:
                hedge = "Low"
            elif val_denorm <= stats['q60']:
                hedge = "Average"
            elif val_denorm <= stats['q80']:
                hedge = "High"
            else:
                hedge = "Very High"
            
            rule_parts.append(f"{feat} is {hedge} (approximately {val_denorm:.1f})")
        
        condition = "\n        AND ".join(rule_parts)
        print(f"\nRULE {i+1}: IF {condition}")
        print(f"        THEN Beer belongs to ARCHETYPE {i+1}")

def validate_rules_efficiency(beer_profiles, centroids, sigmas, U_fcm, N, C):    
    U_rules = np.zeros((C, N))
    for j in range(N):
        U_rules[:, j] = get_beer_membership_adaptive(j, beer_profiles, centroids, sigmas)
    
    # Mean Absolute Error - MAE
    mae = np.mean(np.abs(U_fcm - U_rules))
    
    # correlation
    correlation = np.corrcoef(U_fcm.flatten(), U_rules.flatten())[0, 1]
    
    print("\n=== MODEL VALIDATION REPORT ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Model Fidelity (Correlation): {correlation * 100:.2f}%")
    
    if mae < 0.15:
        print("STATUS: SUCCESS. The fuzzy rules accurately approximate the FCM model.")
    else:
        print("STATUS: WARNING. High discrepancy. Consider adjusting 'sigmas'.")
        
    return U_rules

def validate_styles(full_df, U, C):
    for i in range(C):
        full_df[f'Archetype_{i+1}'] = U[i]
    
    style_analysis = full_df.groupby('Style')[[f'Archetype_{i+1}' for i in range(C)]].mean()
    
    print("\n--- BEER STYLES ANALYSIS (Mapping Style to Archetype) ---")
    for i in range(C):
        print("="*60, end="")
        column = f'Archetype_{i+1}'
        top_styles = style_analysis[column].sort_values(ascending=False).head(C)
        print(f"\nThe following styles are representative of Archetype {i+1}:")
        print(top_styles.round(4).to_string())
        print()
    
    return style_analysis

def print_membership_values_matrix_vs_rules(U, C, full_df, beer_profiles, centroids, sigmas):
    print(f"{'Name':<30} | {'Style':<25} | {'FCM U':<10} | {'Rule Score':<10} | {'Score Difference'}")
    print("-" * 103)
    for i in range(C):
        top_indices = np.argsort(U[i])[-3:][::-1] # type: ignore
        print(f"\n>>> АNALYSIS FOR CLUSTER {i+1} <<<")
        for idx in top_indices:
            beer_row = full_df.iloc[idx]
            
            fcm_val = U[i,idx] #type: ignore
            rule_vals = get_beer_membership_adaptive(idx, beer_profiles, centroids, sigmas)
            rule_val = rule_vals[i]
            diff = np.absolute(fcm_val - rule_val)
            print(f"{str(beer_row['Name'])[:30]:<30} | {str(beer_row['Style'])[:25]:<25} | {fcm_val:.4f}     | {rule_val:.4f}     | {diff:.4f}")

#
def print_archetype_definitions_denorm(features, centroids, sigmas, original_df):
    print("=== DENORMALIZED DEFINITIONS ===")
    for i in range(len(centroids)):
        print(f"\nARCHETYPE {i+1}:")
        for j, feat in enumerate(features):
            f_min = original_df[feat].min()
            f_max = original_df[feat].max()
            
            # denormalization
            actual_center = centroids[i, j] * (f_max - f_min) + f_min
            actual_sigma = sigmas[i, j] * (f_max - f_min)
            
            print(f"  - {feat:12}: Center {actual_center:.2f} (±{actual_sigma:.2f})")

def print_defining_features(features, sigmas):
    print("\n" + "="*60)
    print("ARCHETYPE'S DEFINING FEATURES")
    print("="*60)
    
    for i in range(sigmas.shape[0]):
        # smallest sigmas first
        strict_indices = np.argsort(sigmas[i])
        
        print(f"\nArchetype {i+1} is mainly defined by the following characteristics:")
        for rank in range(3): 
            idx = strict_indices[rank]
            feature = features[idx]
            sigma_val = sigmas[i, idx]
            print(f"  {rank+1}. {feature:<15} (Sigma: {sigma_val:.4f})")

if __name__ == "__main__":
    pass