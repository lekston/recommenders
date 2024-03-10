import numpy as np
import pandas as pd
import sys
import time

from scipy import sparse
from scipy.stats import pearsonr


# NOTE: all sparse matrices / arrays use indexing that starts with 1 (userId & movieId)
# work-around:
#   userId = 0 is a dummy user
#   movieId = 0 is a dummy movie






def read_data(filepath: str) -> pd.DataFrame:
    start_sec = time.time()
    dataset = pd.read_csv(
        filepath,
        header=0,
        usecols=["userId","movieId","rating",]
    )
    end_sec = time.time()
    print(f'Read: {len(dataset)}. It took {end_sec - start_sec}')

    assert all([column_name in dataset.columns for column_name in ["userId", "movieId", "rating"]]), \
        "Incorrect file format"

    # import pdb; pdb.set_trace()
    return dataset


def build_rating_matrix() -> sparse.coo_matrix:
    '''
        rating_matrix format:
        - row index is userId
        - column index is movieId
    '''
    data = read_data('./movielens-20m-dataset/rating.csv')

    user_count = data['userId'].max()
    movie_count = data['movieId'].max()
    print(f'Found reviews from {user_count} users and for {movie_count} movies')

    data_i_j = (data['rating'], (data['userId'], data["movieId"]))

    rating_matrix = sparse.coo_matrix(data_i_j, shape=(user_count + 1, movie_count + 1))

    return rating_matrix



# TODO: train, dev, test split

# Recommend:
# 1. Find average ratings by each user & expres each rating as a deviation
# 2. Calculate user-user weights:
#   - for each user precompute an ordered dict storing correlations to the top-k other most
#       similar/dissimilar users (in absolute terms)
#   - store this as csv


if __name__ == '__main__':
    rating_matrix: sparse.csr_matrix = build_rating_matrix().tocsr() # Compressed sparse row for efficient compute
    rating_matrix.sort_indices() # in-place
    sys.stdout.write("Calculating user biases...")
    number_of_entries_per_user = np.expand_dims(rating_matrix.getnnz(axis=1), axis=1)
    number_of_entries_per_user[0] = 1 # dummy user
    per_user_bias = rating_matrix.sum(axis=1) / number_of_entries_per_user
    print("Done")
    import pdb; pdb.set_trace()

    sys.stdout.write("Centering user ratings...")
    total_count = rating_matrix.getnnz()
    data_indices_indptr = (rating_matrix.data - per_user_bias, rating_matrix.indices, range(0, total_count))
    rating_matrix_centered = sparse.csr_matrix(data_indices_indptr)
    print("Done")
    import pdb; pdb.set_trace()