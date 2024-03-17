from __future__ import annotations

import math
import numpy as np
import pandas as pd
import sys
import time

from sortedcollections import SortedList

from scipy import sparse
from scipy.stats import pearsonr
from scipy.stats._result_classes import PearsonRResult
from typing import List, NamedTuple, Set, Tuple

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
# 1. Find average ratings by each user & express each rating as a deviation - DONE
# 2. Calculate user-user weights:
#   - for each user precompute an ordered dict storing correlations to the top-k other most
#       similar/dissimilar users (in absolute terms)
#   - store this as csv
# 3. Compute predicted score
# 4. Test on 'held-out' validation
#       > select some random users and remove random ratings from them
#       > re-calculate user-user weights between validation set and 'train' set
#       > predict *known* gold truth values of ratings for validation users

'''
Performance:
    Total relations: 16.9*1e9 !!!
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
    Option for speeding up: multiprocessing parallel for
    - parallelize over blocks of traingular matrix
    - gather all correlation values
    - insert all results into the top_k_u2u shortlist (single threaded)
    TODO:
    - consider cosine instead of pearsonr as data is already centered (but I would still need the p-value)
'''

class DummyPearsonr(NamedTuple):
    statistic: float
    pvalue: float


class CorrelationEntry(NamedTuple):
    correlation: float
    user_id: int


class UserUserWeights(object):
    SingleUserTopKSortedEntries = SortedList[CorrelationEntry]

    def __init__(self, num_of_users: int, top_k: int = 20, absolute_sort: bool = True):
        '''
        if absolute_sort == True: store both the most similar and the most dissimilar users
        '''
        super().__init__()
        assert num_of_users > 0  and top_k > 0
        self._num_of_users = 0
        self._sorting_key = (lambda x: -math.fabs(x[0])) if absolute_sort else (lambda x: -x[0])
        self._overlap_abs_min = 25
        self._overlap_ratio = 0.5
        self._max_p_value = 0.05
        self._sampling_ratio = 0.04
        self._top_k = top_k
        self._top_k_u2u_correlations: List[self.SingleUserTopKSortedEntries] = [
            SortedList(key=self._sorting_key) for _ in range(0, num_of_users)
        ]

    def __getitem__(self, index) -> SingleUserTopKSortedEntries:
        return self._top_k_u2u_correlations[index]

    def load_from_matrix(self, centered_rating_matrix: sparse.csr_matrix, shard_id=0, shards_count=1):
        pid = os.getpid()
        print(f'process id: {pid} processing shard {shard_id} out of {shards_count}')
        assert 0 < shard_id + 1 <= shards_count
        # def dummy_pearsonr_algo(x, y):
        #     return DummyPearsonr(statistic=random.uniform(-1.0, 1.0), pvalue=random.uniform(0.3,0.8))
        # pearsonr_algo = dummy_pearsonr_algo
        pearsonr_algo = pearsonr
        def _calculate_corr_ref(user_i:sparse.csr_matrix, user_j:sparse.csr_matrix , common_item_indices: Set):
            '''
            Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
            '''
            user_i_rating_shortlist = np.empty(len(common_item_indices))
            user_j_rating_shortlist = np.empty(len(common_item_indices))
            for out_idx, index_value in enumerate(common_item_indices):
                user_i_rating_shortlist[out_idx] = user_i.data[user_i.indices.searchsorted(index_value)]
                user_j_rating_shortlist[out_idx] = user_j.data[user_j.indices.searchsorted(index_value)]
            return pearsonr_algo(user_i_rating_shortlist, user_j_rating_shortlist)
        def _calculate_corr_array_like(user_i:sparse.csr_matrix, user_j:sparse.csr_matrix , common_item_indices: Set):
            '''
            Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
            '''
            common_item_indices = sorted(common_item_indices)
            user_i_csr_indices = user_i.indices.searchsorted(common_item_indices)
            user_j_csr_indices = user_j.indices.searchsorted(common_item_indices)
            return pearsonr_algo(user_i.data[user_i_csr_indices], user_j.data[user_j_csr_indices])

        start_time = time.time()
        num_updated = 0
        for user_i_idx in range(1, centered_rating_matrix.shape[0]):
            user_i = centered_rating_matrix.getrow(user_i_idx)
            lower_j_idx = 1 + int((shard_id)/shards_count * user_i_idx)
            upper_j_idx = int((shard_id+1)/shards_count * user_i_idx)
            # poor-man's Monte-Carlo selection
            size_of_subset = int(self._sampling_ratio*(upper_j_idx - lower_j_idx))
            selection_subset = np.unique(np.random.uniform(lower_j_idx, upper_j_idx, size_of_subset).astype(int))
            # NOTE: to use the entire domain go for: selection_subset = range(lower_j_idx, upper_j_idx)
            for user_j_idx in selection_subset:
                user_j = centered_rating_matrix.getrow(user_j_idx)
                common_item_indices = set(user_i.indices).intersection(set(user_j.indices))
                required_overlap = min(len(user_i.indices), len(user_j.indices)) * self._overlap_ratio
                required_overlap = max(required_overlap, self._overlap_abs_min)
                if len(common_item_indices) > required_overlap:
                    # process users with high overlap
                    # corr_value: PearsonRResult = _calculate_corr_ref(user_i, user_j, common_item_indices)
                    corr_value: PearsonRResult = _calculate_corr_array_like(user_i, user_j, common_item_indices)
                    if corr_value.pvalue > self._max_p_value:
                        self.add_entry((user_i_idx, user_j_idx), corr_value.statistic)
                        num_updated += 1
                else:
                    if len(common_item_indices) > self._overlap_abs_min:
                        # TODO: store some users with medium overlap to get back to them if needed
                        # TODO: potentially sort/cluster users by most movie overlap
                        #       (eval overlap for each user combination)
                        pass
                num_of_explored_relations = int(0.5 * user_i_idx * (user_i_idx - 1) + user_j_idx)
                if num_of_explored_relations % 10000 == 0:
                    print(f"[{shard_id+1}/{shards_count}] Elapsed time: {time.time() - start_time:.4f}\t"
                          f"Explored: {num_of_explored_relations}\tUpdated: {num_updated} (i,j): {user_i_idx, user_j_idx}")

    def add_entry(self, two_users: Tuple[int, int], corr_value: int):
        user_i, user_j = two_users
        def _update_user(user, other_user, value):
            user_top_k_entries: self.SingleUserTopKSortedEntries = self._top_k_u2u_correlations[user]
            user_top_k_entries.add((value, other_user))
            if len(user_top_k_entries) > self._top_k:
                user_top_k_entries.pop()
        
        _update_user(user_i, user_j, corr_value)
        _update_user(user_j, user_i, corr_value)

    @property
    def top_k_u2u_correlations(self):
        return self._top_k_u2u_correlations # TODO: convert to dataframe??

    def serialize(self, filename: str):
        # import pdb; pdb.set_trace()
        total_written = 0
        u2u_corr_list = [(idx, l) for idx, l in enumerate(self._top_k_u2u_correlations) if len(l) > 0]
        with open(filename, 'w') as ofile:
            for idx, row in u2u_corr_list:
                output = [str(entry) for entry in row]
                ofile.write(f'{ idx},\t' + ', '.join(output) + '\n')
                total_written += 1
                if total_written % 10000 == 0:
                    print(f"Printed {total_written} lines")
        # import pdb; pdb.set_trace()
        # pass
    
    @staticmethod
    def build_from_file(filename: str) -> UserUserWeights:
        # TODO: TESTME
        import pdb; pdb.set_trace()
        data = pd.read_csv(filename) # TODO
        u2u_weights = UserUserWeights(num_of_users=1)
        u2u_weights._num_of_users = data.shape[0]
        u2u_weights._top_k = data.shape[1] # or max len of the longest row
        u2u_weights._top_k_u2u_correlations = data # TODO: is it worth converting back to list of sorted lists??
        return u2u_weights


from multiprocessing import Pool
import os


if __name__ == '__main__':
    rating_matrix: sparse.csr_matrix = build_rating_matrix().tocsr() # Compressed sparse row for efficient compute
    rating_matrix.sort_indices() # in-place

    sys.stdout.write("Calculating user biases...")
    number_of_entries_per_user = np.expand_dims(rating_matrix.getnnz(axis=1), axis=1)
    number_of_entries_per_user[0] = 1 # dummy user
    per_user_bias = rating_matrix.sum(axis=1) / number_of_entries_per_user
    print("Done")

    total_count = rating_matrix.getnnz()
    def _subtract_scalar_from_csr_data(original_data: sparse.csr_matrix, per_user_bias: np.array) -> np.ndarray:
        '''
        This implements sparse subtraction of scalars
        TODO: sparse addition & subtraction of scalars should be moved to a dedicated module
        '''
        per_row_count = original_data.getnnz(axis=1)
        new_data = np.zeros((total_count,))
        offset = 0
        index_ptr = [0]
        for row, nnz in zip(range(0, original_data.shape[0]), per_row_count):
            index_ptr.append(offset)
            new_data[offset:offset+nnz] = original_data.getrow(row).data - per_user_bias[row]
            offset += nnz

        data_indices_indptr = (new_data, original_data.indices, index_ptr)
        rating_matrix_centered = sparse.csr_matrix(data_indices_indptr, shape=rating_matrix.shape)
        return rating_matrix_centered

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = _subtract_scalar_from_csr_data(rating_matrix, per_user_bias)
    print("Done")

    print(f"Time spend on bias removal: {time.time() - start_time}")

    sys.stdout.write("Computing user-user correlations...")

    def parallelized_user_user(rmat_centered_with_shard_details):
        rmat_centered, shard_id, shards_count = rmat_centered_with_shard_details
        u2u_w = UserUserWeights(num_of_users=rmat_centered.shape[0], top_k=10)
        u2u_w.load_from_matrix(rmat_centered, shard_id, shards_count)
        return u2u_w

    start_time = time.time()
    pool_size = 8
    if pool_size > 1:
        with Pool(pool_size) as p:
            u2u_weights_to_merge = p.map(
                parallelized_user_user,
                [(rating_matrix_centered, i, pool_size) for i in range(0, pool_size)]
            )
    else:
        u2u_weights_to_merge = [parallelized_user_user((rating_matrix_centered, 0, 32))]

    [u2u_w.serialize(f"u2u_demo_{idx}.csv") for idx, u2u_w in enumerate(u2u_weights_to_merge)]

    print("Done")
    print(f"Time spend on user-user correlations: {time.time() - start_time}")
