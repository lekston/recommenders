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
# 3. Implement predicted score computation
# 4. Test on 'held-out' validation
#       > select some random users and remove random ratings from them
#       > re-calculate user-user weights between validation set and 'train' set
#       > predict *known* gold truth values of ratings for validation users

class CorrelationEntry(NamedTuple):
    correlation: float
    user_id: int


class UserUserWeights(object):
    SingleUserTopKSortedEntries = SortedList[CorrelationEntry]

    def __init__(self, num_of_users: int, top_k: int = 20, absolute_sort: bool = True):
        '''
        if abosulte_sort == True: store both the most similar and the most dissimilar users
        '''
        super().__init__()
        assert num_of_users > 0  and top_k > 0
        self._num_of_users = 0
        self._sorting_key = (lambda x: -math.fabs(x[0])) if absolute_sort else (lambda x: -x[0])
        self._min_overlap = 20
        self._min_p_value = 0.5
        self._top_k = top_k
        self._top_k_u2u_correlations: List[self.SingleUserTopKSortedEntries] = [
            SortedList(key=self._sorting_key) for _ in range(0, num_of_users)
        ]

    def __getitem__(self, index) -> SingleUserTopKSortedEntries:
        return self._top_k_u2u_correlations[index]

    def load_from_matrix(self, centered_rating_matrix: sparse.csr_matrix):
        def _calculate_corr(user_i:sparse.csr_matrix, user_j:sparse.csr_matrix , common_item_indices: Set):
            '''
            Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
            '''
            user_i_rating_shortlist = []
            user_j_rating_shortlist = []
            for item_idx in common_item_indices:
                user_i_rating_shortlist.append(user_i.data[user_i.indices.searchsorted(item_idx)])
                user_j_rating_shortlist.append(user_j.data[user_j.indices.searchsorted(item_idx)])
            return pearsonr(user_i_rating_shortlist, user_j_rating_shortlist)
        start_time = time.time()
        num_updated = 0
        # TODO: option for speeding up: multiprocessing parallel for
        # - parallelize over blocks of traingular matrix
        # - gather all correlation values
        # - insert all results into the top_k_u2u shortlist (single threaded)
        for user_i_idx in range(1, centered_rating_matrix.shape[0]):
            user_i = centered_rating_matrix.getrow(user_i_idx)
            for user_j_idx in range(1, user_i_idx):
                user_j = centered_rating_matrix.getrow(user_j_idx)
                common_item_indices = set(user_i.indices).intersection(set(user_j.indices))
                if len(common_item_indices) > self._min_overlap:
                    corr_value: PearsonRResult = _calculate_corr(user_i, user_j, common_item_indices)
                    if corr_value.pvalue < self._min_p_value:
                        self.add_entry((user_i_idx, user_j_idx), corr_value.statistic)
                        num_updated += 1
                num_of_explored_relations = int(0.5 * user_i_idx * (user_i_idx - 1) + user_j_idx)
                if num_of_explored_relations % 10000 == 0:
                    print(f"Elapsed time: {time.time() - start_time:.4f}\t"
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
        import pdb; pdb.set_trace()
        total_written = 0
        with open(filename, 'r') as ofile:
            for row in self._top_k_u2u_correlations:
                output = [entry for entry in row]
                ofile.write(','.join(output) + '\n')
                total_written += 1
                if total_written % 10000 == 0:
                    print(f"Printed {total_written} lines")
        import pdb; pdb.set_trace()
        pass
    
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
    start_time = time.time()
    u2u_w = UserUserWeights(num_of_users=rating_matrix_centered.shape[0], top_k=10)
    u2u_w.load_from_matrix(rating_matrix_centered)
    print("Done")
    print(f"Time spend on user-user correlations: {time.time() - start_time}")

    u2u_w.serialize("u2u_demo.csv")