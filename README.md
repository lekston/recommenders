# Recommender system coding examples
Some basic examples of recommender system implementations.

Contents:
- Collaborative Filtering example with multi-core processing
- ... (WIP)

# Collaborative Filtering

Following this approach recommendations are prepared in the following manner:
1. Find average ratings by each user & express each rating as a deviation
2. Calculate user-user weights (takes time!)
  - for each user precompute an ordered dict storing correlations to the top-k other most
      similar/dissimilar users (in absolute terms)
  - store this as csv (this can be re-used for requests)
3. Compute predicted score
  - get all users NB_u(i) correlated with user i
  - extract all movies Omega_NB_u(i) rated by users NB_u(i)
  - calculate scores for all movies (Omega_NB_u(i) \ Omega_u(i)) not seen by user i
  - recommend the best match
4. Optional: test on 'held-out' validation
      > Naive: predict *known* gold truth values of ratings for random users from the train set
      > TODO: train, dev, test split
      > select some random users and remove random ratings from them
      > re-calculate user-user weights between validation set and 'train' set
      > Desired: predict *known* gold truth values of ratings for validation users


# Dataset:
To get started please download the
https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
and extract it to folder `./movielens-20m-dataset/`

# Notes on parallelization
Brute-force multi-processing:
 - rating matrix is loaded and preprocessed by the main process
 - parallelization is done over sub-triangles of the traingular matrix
 - correlation values from all workers are then merged into a single UserUser matrix (again single threaded)

# Complexity and performance
Collaborative filtering requires User-User weight matrix generation. Current approach to this has O(n^2) time complexity is the main computational bottleneck.

Some performance runs exploring the implications of this are documented below.

Environment: Ubuntu, CPU: i7-8565U, RAM: 16GB

Performance:
    Total relations: 16.9*1e9 !!! (i.e. all user-user combinations)
    - min overlap 20, min p_value 0.5
        Single core: 2.8*1e5 per 1min -> 16.8*1e6 per 1h -> 1000h single core
    - min overlap 25, min p_value 0.35
        Single core: 3.5*1e5 per 1min -> 21.0*1e6 per 1h -> 800h single core
        Multi-core 4x (without merging): 8.3*1e5 -> 49.8*1e6
        Multi-core 8x (4x physical) (without merging): 11.0*1e5 -> 66.0*1e6
        Expected time using basic parallelization: 256h!!!! (11days)
        Mutli-core 8x (without merging; searchsorted array like): -> 11.0*1e5 (no change)
    - min overlap max(25, min(movies_u_i, movies_u_j)-10), min p_value 0.35
        Mutli-core 8x (without merging; searchsorted array-like): -> 24.0*1e5 (no change)

    - min overlap 25, min p_value 0.35 (no storage cost) -> same CPU runtime
    - min overlap 25, min p_value 0.35 (dummy Pearson, skipping searchsorted)
        Single core: 8.6*1e5 per 1min -> 51.6*1e6 per 1h -> pearsonr + searchsorted eat up 60% of CPU
    - min overlap 25, min p_value 0.35 (dummy Pearson, using searchsorted on sets)
        Single core: 6.6*1e5 per 1min -> 51.6*1e6 per 1h -> pearsonr alone uses ~25% of CPU
TODO:
- consider cosine instead of pearsonr as data is already centered (but I would still need the p-value)


