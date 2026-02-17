import numpy as np

MAX_RANDOM_RESTARTS = 50
EPSILON = 0.005
MAX_ITERATIONS = 1000

def calculate_objective_function(X, U, centroids, m):
    # X: (N, 7), U: (C, N), centroids: (C, 7)
    diff = centroids[:, None, :] - X[None, :, :]
    D_sq = np.sum(diff**2, axis=2) # ||x_i - c_j||^2
    
    Um = U ** m

    J_m = np.sum(Um * D_sq)
    return J_m

def update_centroids(X, U, m):
    # X -> (N, 7), U -> (C, N)
    Um = U ** m  # shape (C, N)
    
    # (C, N) @ (N, 7) ->  (C, 7)
    numerator = Um @ X
    
    denominator = Um.sum(axis=1, keepdims=True) # shape (C, 1)
    
    centroids = numerator / denominator
    return centroids # shape (C, 7)

def update_U(centroids, beer_profiles, m):
    # centroids: (C, 7)
    # beer_profiles: (N, 7)
    
    # distances between all beers and all centroids
    # broadcasting: (C, 1, 7) - (1, N, 7) -> (C, N, 7)
    diff = centroids[:, None, :] - beer_profiles[None, :, :]
    D = np.linalg.norm(diff, axis=2) # shape (C, N)
    
    # to avoid division by zero
    D = np.fmax(D, 1e-9)
    
    # u_ij = 1 / sum( (d_ij / d_kj) ^ (2/(m-1)) )
    power = 2 / (m - 1)
    D_inv = 1 / (D ** power) # (C, N)
    
    D_inv_sum = D_inv.sum(axis=0, keepdims=True) # (1, N)
    
    U_new = D_inv / D_inv_sum # (C, N)
    return U_new

def fuzzy_c_means_random_restart(beer_profiles, C, N, m, print_state = False):
    best_U = None
    best_centroids = None
    min_cost = float('inf')

    if print_state:
        print("\nRandom restarts in progress...")

    for _ in range(MAX_RANDOM_RESTARTS): 
        current_centroids, current_U = fuzzy_c_means_single(beer_profiles, C, N, m)
        current_cost = calculate_objective_function(beer_profiles, current_U, current_centroids, m)
        if current_cost < min_cost:
            min_cost = current_cost
            best_U = current_U
            best_centroids = current_centroids
    if print_state:
        print("\nDone with random restarts.")
    return best_centroids, best_U

def fuzzy_c_means_single(beer_profiles, C, N, m, print_state = False):
    # should consider smart initialization

    # membership matrix (C, N)
    # for each archetype, for each beer -> degree of membership
    U = np.random.rand(C, N)
    U = U / U.sum(axis=0, keepdims=True)

    # main part
    iteration = 0
    while True:
        iteration += 1
        U_old = U.copy()  
        centroids = update_centroids(beer_profiles, U, m)   
        U = update_U(centroids, beer_profiles, m)
        
        diff_norm = np.linalg.norm(U - U_old)
        
        if print_state:
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Change in U = {diff_norm:.6f}")
            
        if diff_norm < EPSILON or iteration > MAX_ITERATIONS:
            if print_state:
                print(f"\nConvergence reached at iteration {iteration}!")
            break

    return centroids, U

if __name__ == "__main__":
    pass
