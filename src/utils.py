import numpy as np
import pandas as pd
import sys
import time

from enum import Enum
from multiprocessing import Pool

from scipy import sparse
from typing import List, Optional

from .cf_weights import (
    CollaborativeFilteringWeights
)


class CFilterType(Enum):
    user2user: str = 'user2user'
    item2item: str = 'item2item'


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

    return dataset


def build_rating_matrix(dataset: str = './movielens-20m-dataset/rating.csv') -> sparse.coo_matrix:
    '''
        rating_matrix format:
        - row index is userId
        - column index is movieId
    '''
    data = read_data(dataset)

    user_count = data['userId'].max()
    movie_count = data['movieId'].max()
    print(f'Found reviews from {user_count} users and for {movie_count} movies')

    data_i_j = (data['rating'], (data['userId'], data["movieId"]))

    rating_matrix = sparse.coo_matrix(data_i_j, shape=(user_count + 1, movie_count + 1))

    return rating_matrix


def unbias_the_ratings(ratings_matrix: sparse.coo_matrix) -> sparse.csr_matrix:
    rating_matrix: sparse.csr_matrix = ratings_matrix.tocsr() # Compressed sparse row for efficient compute
    rating_matrix.sort_indices() # in-place

    sys.stdout.write("Calculating user biases...")
    number_of_entries_per_user = np.expand_dims(rating_matrix.getnnz(axis=1), axis=1)
    number_of_entries_per_user[0] = 1 # dummy user
    per_user_bias = rating_matrix.sum(axis=1) / number_of_entries_per_user
    print("Done")

    total_count = rating_matrix.getnnz()
    def _subtract_scalar_from_csr_data(original_data: sparse.csr_matrix, per_user_bias: np.array) -> sparse.csr_matrix:
        '''
        This implements sparse subtraction of scalars
        TODO: sparse addition & subtraction of scalars should be moved to a dedicated module
        '''
        per_row_non_zero_count = original_data.getnnz(axis=1)
        new_data = np.zeros((total_count,))
        offset = 0
        index_ptr = [0]
        for row, nnz in zip(range(0, original_data.shape[0]), per_row_non_zero_count):
            index_ptr.append(offset)
            new_data[offset:offset+nnz] = original_data.getrow(row).data - per_user_bias[row]
            offset += nnz

        data_indices_indptr = (new_data, original_data.indices, index_ptr)
        rating_matrix_centered = sparse.csr_matrix(data_indices_indptr, shape=rating_matrix.shape)
        return rating_matrix_centered

    return _subtract_scalar_from_csr_data(rating_matrix, per_user_bias)


def _parallelized_cf_worker(rmat_centered_with_shard_details) -> List[CollaborativeFilteringWeights.TopKSortedNeighbors]:
    rmat_centered, shard_id, shards_count = rmat_centered_with_shard_details
    print(f"{rmat_centered_with_shard_details}")
    # TODO: support selectable sampling_ratio (0.04 should be default for u2u, 0.2 for i2i)
    u2u_w = CollaborativeFilteringWeights(num_of_users=rmat_centered.shape[0], top_k=10, sampling_ratio=0.2)
    u2u_w.load_from_matrix(rmat_centered, shard_id, shards_count)
    return u2u_w.get_correlations_as_list()


def calc_collaborative_filtering_weights(
        pool_size: int = 8,
        merged_output_file: Optional[str] = None,
        run_type: CFilterType = CFilterType.user2user
    ) -> None:

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    coo_ratings_matrix = build_rating_matrix()
    if run_type == CFilterType.item2item:
        coo_ratings_matrix = coo_ratings_matrix.transpose()
    rating_matrix_centered = unbias_the_ratings(coo_ratings_matrix)
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    if run_type == CFilterType.user2user:
        sys.stdout.write("Computing user-user correlations...")
        file_prefix = 'u2u'
    else:
        sys.stdout.write("Computing item-item correlations...")
        file_prefix = 'i2i'

    start_time = time.time()
    if pool_size > 1:
        with Pool(pool_size) as p:
            u2u_weights_to_merge = p.map(
                _parallelized_cf_worker,
                [(rating_matrix_centered, i, pool_size) for i in range(0, pool_size)]
            )
    else:
        u2u_weights_to_merge = [_parallelized_cf_worker((rating_matrix_centered, 0, 32))]

    [
        CollaborativeFilteringWeights.serialize(u2u_w, f"{file_prefix}_weights_part_{idx}.csv")
        for idx, u2u_w in enumerate(u2u_weights_to_merge)
    ]

    print("Done")
    print(f"Time spend on calculating correlations: {time.time() - start_time}")

    # TODO: configurable option to merge without storing
    if merged_output_file is not None:
        print("Merging")
        u2u_merged = merge_parts([
            f"{file_prefix}_weights_part_{idx}.csv" for idx in range(0, pool_size)
        ], merged_output_file)
        print("Merging Done")


def merge_parts(files: List[str], output_fname: str = "merged.csv") -> CollaborativeFilteringWeights:
    u2u_merged = CollaborativeFilteringWeights.build_from_files(files)
    all_lengths = [(idx, len(entry)) for idx, entry in enumerate(u2u_merged.get_correlations_as_list())]
    print(f"Merged list contains: {len(u2u_merged)} entries")
    print(f"Including: {len(['' for idx, entry_len in all_lengths if entry_len > 0])} non-zero entries")
    CollaborativeFilteringWeights.serialize(u2u_merged.get_correlations(), output_fname)
    return u2u_merged
