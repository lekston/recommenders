# Recommender system coding examples
Some basic examples of recommender system implementations.

Contents:
- User-User Collaborative Filtering example with multi-core processing
- Item-Item Collaborative Filtering (...WIP)

# User-User Collaborative Filtering

Following this approach recommendations are prepared in the following manner:
1. Find average ratings by each user & express each rating as a deviation
2. Calculate user-user weights (takes time!)
  - for each user precompute an ordered dict storing correlations to the top-k other most
      similar/dissimilar users (in absolute terms)
  - store this as csv (this can be re-used for requests)
3. Compute predicted score
  - get all users NB_u(i) correlated with user i
  - select all movies Psi_NB_u(i) rated by users NB_u(i)
  - calculate scores for all movies (Psi_NB_u(i) \ Psi_u(i)) not seen by user i
  - recommend the best match
4. Tests:
  - Naive: predict *known* gold truth values of ratings for random users from the train set
  - TODO: tests on 'held-out' validation:
    - prepare train, dev, test split
    - select some random users and remove random ratings from them
    - re-calculate user-user weights between validation set and 'train' set
    - Desired: predict *known* gold truth values of ratings for validation users

# Item-Item Collaborative Filtering

Following this approach recommendations are prepared in the following manner:
1. Find average ratings of each item (i.e. by all users) & express each rating as a deviation
2. Calculate item-item weights (takes time!)
  - for each item precompute an ordered dict storing correlations to the top-k other most
      similar/dissimilar items (in absolute terms)
  - store this as csv (this can be re-used for requests)
3. Compute predicted score
  - get all items j \subset Psi_u(i) rated by user i
  - select all items NB_i(j) which are correlated with items j \subset Psi_u(i)
  - calculate scores for all items (NB_i(j) \setminus Psi_u(i)) not seen by user i
    - each item from NB_i(j) may be similar / dissimilar to several movies rated by the user i
    - score for items from NB_i(j) is calculated by weighted sum over all similar/dissimilar items j' \subset Psi_u(i)
    - weights come from the item-item weights
    - deviations come from the ratings of the current user only
  - recommend the best match


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
- Total relations: 16.9*1e9! (i.e. all user-user combinations)
- User-2-user runs:
    - min overlap 20, min p_value 0.5
        - Single core: `2.8*1e5` per 1min -> `16.8*1e6` per 1h -> 1000h single core
    - min overlap 25, min p_value 0.35
        - Single core: `3.5*1e5` per 1min -> `21.0*1e6` per 1h -> 800h single core
        - Multi-core 4x (without merging): `8.3*1e5` -> `49.8*1e6` per h
        - Multi-core 8x (4x physical) (without merging): `11.0*1e5` -> `66.0*1e6` per h
        - Expected time using basic parallelization: 256h!!!! (11days)
        - Mutli-core 8x (without merging; searchsorted array like): -> `11.0*1e5` per min (no change)
    - min overlap max(25, min(movies_u_i, movies_u_j)-10), min p_value 0.35
        - Mutli-core 8x (without merging; searchsorted array-like): -> `12.0*1e5` per min
            - sampling ratio 0.04 -> takes ~5 hours on 4 core i7
    - min overlap 25, min p_value 0.35 (no storage cost) -> same CPU runtime
    - min overlap 25, min p_value 0.35 (dummy Pearson, skipping searchsorted)
        - Single core: `8.6*1e5` per 1min -> `51.6*1e6` per 1h -> pearsonr + searchsorted eat up 60% of CPU
    - min overlap 25, min p_value 0.35 (dummy Pearson, using searchsorted on sets)
        - Single core: `6.6*1e5` per 1min -> `51.6*1e6` per 1h -> pearsonr alone uses ~25% of CPU
- Item-2-Item runs
    - min overlap max(25, min(users_m_i, users_m_j)-10), min p_value 0.35
        - Mutli-core 8x (without merging; searchsorted array-like): -> `9.0*1e4` per min
        - (first minute is 10x slower than similar U2U run!!)
        - sampling ratio 0.2
            -  `6.5*1e6` per min -> `0.39*1e9` per h
TODO:
- consider cosine instead of pearsonr as data is already centered (but I would still need the p-value)


