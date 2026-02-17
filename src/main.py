import numpy as np
import src.data_loader as dl
import src.fuzzy_c_means as fcm
import src.fuzzy_rules as fr
import src.visualization as vis

FILE_PATH = "data/beer_profile_and_ratings.csv"
NUMBER_OF_ARCHETYPES = 5
FUZZIFIER = 2 # the m parameter

def main():
    # 1. Loading
    print("\nScript in progress...")
    print("\nReading data in progress...")
    beer_profiles, full_df, features = dl.load_and_normalize(FILE_PATH)
    print("\nDone reading data.")

    # variables for algorithms
    m = FUZZIFIER
    N = beer_profiles.shape[0]
    C = NUMBER_OF_ARCHETYPES

    # 2. Clustering
    centroids, U = fcm.fuzzy_c_means_random_restart(beer_profiles, C, N, m, True)

    # order by ABV, so the cluster order isn't totally random
    sort_indices = np.argsort(centroids[:, 0])  #type:ignore
    centroids = centroids[sort_indices]  #type:ignore
    U = U[sort_indices]   #type:ignore

    # 3. Preparing 
    sigmas = fr.calculate_cluster_sigmas(beer_profiles, U, centroids, m)
   
    # 4. Visualizing
    # 4.1 PCA
    # vis.plot_archetypes_pca(beer_profiles, centroids, U, C) # displays result of fuzzy c means
    fr.print_membership_values_matrix_vs_rules(U, C, full_df, beer_profiles, centroids, sigmas) # displays membership values from the result of fuzzy c means vs from the rules
    vis.plot_archetypes_pca_with_labels(beer_profiles, centroids, U, C, full_df) # displays result of fuzzy c means but with beers from the dataset

    # 4.2 Rules and Membership Functions Graph
    #fr.print_archetype_definitions_denorm(features, centroids, sigmas, full_df) # explains archetypes similarly to the rules
    fr.generate_linguistic_rules_denorm(features, centroids, C, full_df) # displays the rules linguistically 
    # vis.plot_all_archetypes_comparison_denorm(features, centroids, sigmas, full_df) # displays memberships graphics of archetypes
    vis.plot_all_archetypes_comparison_denorm_quantile(features, centroids, sigmas, full_df) # displays memberships graphics of archetypes with quantile values

    # 4.3 Test if rules are reflecting the membership matrix
    U_rules = fr.validate_rules_efficiency(beer_profiles, centroids, sigmas, U, N, C)

    # 4.4 Style - Archetype Heatmap
    fr.validate_styles(full_df, U_rules, C) # shows styles similarity to archetypes
    vis.plot_style_archetype_heatmap(full_df, U_rules, C) # shows styles similarity to archetypes

    # vis.plot_elbow_method(beer_profiles, N, m) # displays elbow method graph and prints ratings in console
    # fr.print_defining_features(features, sigmas) # displays most strict characteristic limitations
    print("\nDone with script.")

if __name__ == "__main__":
    main()