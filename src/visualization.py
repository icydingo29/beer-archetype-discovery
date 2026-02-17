import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.fuzzy_c_means as fcm
from sklearn.decomposition import PCA

def save_plot(filename, C=None):
    if not os.path.exists('img'):
        os.makedirs('img')
    
    # Генериране на пътя
    if C is not None:
        full_path = f"img/{filename}_C{C}.png"
    else:
        full_path = f"img/{filename}.png"
    
    # Записваме (това автоматично презаписва стария файл, ако съществува)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {full_path}")

def plot_archetypes_pca(beer_profiles, centroids, U, C):
    pca = PCA(n_components=2)
    
    pca.fit(beer_profiles)
    
    beer_profiles_2d = pca.transform(beer_profiles)
    centroids_2d = pca.transform(centroids)

    hard_memberships = np.argmax(U, axis=0) # (N,)

    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.jet(np.linspace(0, 1, C)) #type: ignore
    
    for i in range(C):
        mask = (hard_memberships == i)
        plt.scatter(beer_profiles_2d[mask, 0], beer_profiles_2d[mask, 1], 
                    color=colors[i], label=f'Beers in Archetype {i+1}', 
                    alpha=0.6, s=50, edgecolors='w', linewidth=0.5)

    for i in range(C):
        plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                    marker='X', s=300, color=colors[i], 
                    edgecolors='black', linewidth=2, label=f'Centroid of Archetype {i+1}')
        plt.text(centroids_2d[i, 0] + 0.1, centroids_2d[i, 1] + 0.1, 
                 f'C{i+1}', fontsize=12, weight='bold', color=colors[i])

    plt.title('2D PCA VISUALIZATION OF BEER ARCHETYPES', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot("2d_pca_clusters", C)
    plt.show()

def plot_archetypes_pca_with_labels(beer_profiles, centroids, U, C, full_df): 
    pca = PCA(n_components=2)
    beer_profiles_2d = pca.fit_transform(beer_profiles)
    centroids_2d = pca.transform(centroids)
    hard_memberships = np.argmax(U, axis=0)

    plt.figure(figsize=(14, 10))
    colors = plt.cm.jet(np.linspace(0, 1, C)) #type: ignore

    for i in range(C):
        mask = (hard_memberships == i)
        plt.scatter(beer_profiles_2d[mask, 0], beer_profiles_2d[mask, 1], 
                    color=colors[i], label=f'Beers of Archetype {i+1}', alpha=0.6, s=50) 

        top_indices = np.argsort(U[i])[-3:] 
        for idx in top_indices:
            name = full_df.iloc[idx]['Name']
            plt.scatter(beer_profiles_2d[idx, 0], beer_profiles_2d[idx, 1], 
                        color=colors[i], s=100, edgecolors='black', linewidth=1)
            plt.text(beer_profiles_2d[idx, 0] + 0.05, beer_profiles_2d[idx, 1] + 0.05, 
                     name, fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    for i in range(C):
        plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                    marker='X', s=400, color=colors[i], edgecolors='black', label=f'Archetype {i+1}')

    plt.title('PCA: BEER ARCHETYPES WITH REPRESENTATIVE EXAMPLES', fontsize=16)
    plt.legend()
    save_plot("2d_pca_clusters_with_examples", C)
    plt.show()

def plot_all_archetypes_comparison(features, centroids, sigmas):
    num_features = len(features)
    num_archetypes = centroids.shape[0]
    
    colors = plt.cm.get_cmap('tab10', num_archetypes)
    
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(22, 5), sharey=True)
    fig.suptitle("COMPARISON OF ARCHETYPES' MEMBERSHIP FUNCTIONS", fontsize=18, fontweight='bold')

    x = np.linspace(0, 1, 300)

    for i in range(num_features):
        for j in range(num_archetypes):
            mean = centroids[j, i]
            sigma = sigmas[j, i]
            
            y = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            
            axes[i].plot(x, y, label=f'A{j+1}', color=colors(j), lw=2)
            axes[i].fill_between(x, y, alpha=0.1, color=colors(j))
            
        axes[i].set_title(features[i], fontsize=14)
        axes[i].set_xlabel('Value (0-1)')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            axes[i].set_ylabel('Membership value')
            axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type:ignore
    save_plot("membership_all_clusters", num_archetypes)
    plt.show()

def plot_all_archetypes_comparison_denorm(features, centroids, sigmas, original_df):
    num_features = len(features)
    num_archetypes = centroids.shape[0]
    
    nrows = 2
    ncols = int(np.ceil(num_features / nrows)) 
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    
    axes_flat = axes.flatten()
    
    fig.suptitle("COMPARISON OF ARCHETYPES' MEMBERSHIP FUNCTIONS", fontsize=20, fontweight='bold', y=0.98)

    colors = plt.cm.get_cmap('tab10', num_archetypes)

    for i in range(num_features):
        ax = axes_flat[i]
        feat_name = features[i]
        
        f_min_orig = original_df[feat_name].min()
        f_max_orig = original_df[feat_name].max()
        f_range = f_max_orig - f_min_orig

        starts = centroids[:, i] - 4 * sigmas[:, i]
        ends = centroids[:, i] + 4 * sigmas[:, i]
        plot_min_norm = max(0, np.min(starts))
        plot_max_norm = min(1, np.max(ends))

        x_start_raw = plot_min_norm * f_range + f_min_orig
        x_end_raw = plot_max_norm * f_range + f_min_orig

        x_raw = np.linspace(x_start_raw, x_end_raw, 300)
        x_norm = (x_raw - f_min_orig) / f_range
        
        for j in range(num_archetypes):
            mean_norm = centroids[j, i]
            sigma_norm = sigmas[j, i]
            
            y = np.exp(-0.5 * ((x_norm - mean_norm) / sigma_norm) ** 2)
            
            ax.plot(x_raw, y, label=f'A{j+1}', color=colors(j), lw=2.5)
            ax.fill_between(x_raw, y, alpha=0.1, color=colors(j))
            
        ax.set_title(f"Characteristic: {feat_name}", fontsize=14, fontweight='bold', pad=10)
        # ax.set_xlabel(f'Units: {feat_name}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.locator_params(axis='x', nbins=6) 

        if i % ncols == 0:
            ax.set_ylabel('Membership value', fontsize=12)
        
        if i == 0 or i == ncols:
            ax.legend(loc='upper right', fontsize=10)

    for i in range(num_features, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #type:ignore
    save_plot("membership_all_clusters_denorm", num_archetypes)
    plt.show()

def plot_all_archetypes_comparison_denorm_quantile(features, centroids, sigmas, original_df):
    num_features = len(features)
    num_archetypes = centroids.shape[0]
    
    nrows = 2
    ncols = int(np.ceil(num_features / nrows)) 
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    
    axes_flat = axes.flatten()
    
    fig.suptitle("COMPARISON OF ARCHETYPES' MEMBERSHIP FUNCTIONS WITH QUANTILES", fontsize=20, fontweight='bold', y=0.98)

    colors = plt.cm.get_cmap('tab10', num_archetypes)

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

    for i in range(num_features):
        ax = axes_flat[i]
        feat_name = features[i]
        
        f_min_orig = original_df[feat_name].min()
        f_max_orig = original_df[feat_name].max()
        f_range = f_max_orig - f_min_orig

        starts = centroids[:, i] - 4 * sigmas[:, i]
        ends = centroids[:, i] + 4 * sigmas[:, i]
        plot_min_norm = max(0, np.min(starts))
        plot_max_norm = min(1, np.max(ends))

        x_start_raw = plot_min_norm * f_range + f_min_orig
        x_end_raw = plot_max_norm * f_range + f_min_orig

        x_raw = np.linspace(x_start_raw, x_end_raw, 300)
        x_norm = (x_raw - f_min_orig) / f_range

        q_keys = ['q20', 'q40', 'q60', 'q80']
        for idx, key in enumerate(q_keys):
            q_val = feature_stats[feat_name][key]
            if x_start_raw <= q_val <= x_end_raw:
                text_and_line_color = ''
                if idx % 2 == 0:
                    text_and_line_color = 'red'
                    y_pos = 1.02
                    va_pos = 'bottom'
                else:
                    text_and_line_color = 'blue'
                    y_pos = -0.02
                    va_pos = 'top'
                ax.axvline(q_val, color=text_and_line_color, linestyle=':', alpha=0.5, linewidth=2.4)
                
                ax.text(q_val, y_pos, f'{q_val:.1f}', 
                        transform=ax.get_xaxis_transform(),
                        color=text_and_line_color, fontsize=9, fontweight='bold',
                        ha='center', va=va_pos)
        
        for j in range(num_archetypes):
            mean_norm = centroids[j, i]
            sigma_norm = sigmas[j, i]
            
            y = np.exp(-0.5 * ((x_norm - mean_norm) / sigma_norm) ** 2)
            
            ax.plot(x_raw, y, label=f'A{j+1}', color=colors(j), lw=2.5)
            ax.fill_between(x_raw, y, alpha=0.1, color=colors(j))
            
        ax.set_title(f"{feat_name}", fontsize=14, fontweight='bold', pad=10)
        # ax.set_xlabel(f'Units: {feat_name}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.locator_params(axis='x', nbins=6) 

        if i % ncols == 0:
            ax.set_ylabel('Membership value', fontsize=12)
        
        if i == 0 or i == ncols:
            ax.legend(loc='upper right', fontsize=10)

    for i in range(num_features, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) #type:ignore
    save_plot("membership_all_clusters_denorm_quant", num_archetypes)
    plt.show()

def plot_style_archetype_heatmap(df, U, num_archetypes):
    archetype_names = [f'Archetype {i+1}' for i in range(num_archetypes)]
    
    membership_df = pd.DataFrame(U.T, columns=archetype_names)
    temp_df = pd.concat([df.reset_index(drop=True), membership_df], axis=1)

    style_counts = temp_df['Style'].value_counts()
    popular_styles = style_counts[style_counts > 10].index
    
    heatmap_data = temp_df[temp_df['Style'].isin(popular_styles)]
    heatmap_data = heatmap_data.groupby('Style')[archetype_names].mean()

    plt.figure(figsize=(12, 25))
    sns.heatmap(heatmap_data, 
                annot=True,       
                fmt=".2f",        
                cmap="YlGnBu",    
                cbar_kws={'label': 'Average Membership Value'})

    plt.title('Correlation between Beer Styles and Archetypes', fontsize=16, fontweight='bold')
    plt.xlabel('Archetypes (Fuzzy Clusters)', fontsize=12)
    plt.ylabel('Original Styles (from data)', fontsize=12)
    
    plt.tight_layout()
    save_plot("archetype_style_heatmap", num_archetypes)
    plt.show()

def plot_elbow_method(beer_profiles, N, m):
    print("\nElbow method in progress...")
    counts = range(2, 20)
    obj_func_values = []

    for c in counts:
        current_C, current_U = fcm.fuzzy_c_means_random_restart(beer_profiles, c, N, m)
        current_val = fcm.calculate_objective_function(beer_profiles, current_U, current_C, m)
        obj_func_values.append(current_val)

    plt.figure(figsize=(8, 5))
    plt.plot(counts, obj_func_values, 'bo-')
    plt.xlabel('Number of Archetypes (C)')
    plt.ylabel('Objective Function')
    plt.title('Elbow method for determining the number of archetypes')
    plt.grid(True)
    print("\nDone with elbow method.")
    # print(obj_func_values)
    save_plot("elbow_method")
    plt.show()

if __name__ == "__main__":
    pass