from __future__ import annotations

import math
import numpy as np
import os
import pandas as pd
import sys
import time

from multiprocessing import Pool
from sortedcollections import SortedList

from scipy import sparse
from scipy.stats import pearsonr
from scipy.stats._result_classes import PearsonRResult
from typing import List, NamedTuple, Set, Tuple, TypeAlias


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


def ratings_matrix_unbiased(ratings_matrix: sparse.coo_matrix) -> sparse.csr_matrix:
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



# TODO: train, dev, test split

# Recommend:
# 1. Find average ratings by each user & express each rating as a deviation - DONE
# 2. Calculate user-user weights:
#   - for each user precompute an ordered dict storing correlations to the top-k other most
#       similar/dissimilar users (in absolute terms)
#   - store this as csv
#   DONE!
# 3. Compute predicted score
#   - get all users NB_u(i) correlated with user i
#   - extract all movies Omega_NB_u(i) rated by users NB_u(i)
#   - calculate scores for all movies (Omega_NB_u(i) \ Omega_u(i)) not seen by user i
#   - recommend the best match
# 4. Optional: test on 'held-out' validation
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

    Multiprocessing parallelization"
    DONE - parallelization is done over sub-triagnles of the traingular matrix
    DONE - gather all correlation values
    DONE - insert all results into the top_k_u2u shortlist (single threaded)
    TODO:
    - consider cosine instead of pearsonr as data is already centered (but I would still need the p-value)
'''

class DummyPearsonr(NamedTuple):
    statistic: float
    pvalue: float


class CorrelationEntry(NamedTuple):
    correlation: float
    user_id: int


LOGGING_PERIOD = 20000


class UserUserWeights(object):
    SingleUserTopKSortedEntries : TypeAlias = SortedList[CorrelationEntry]

    def __init__(
            self,
            num_of_users: int,
            top_k: int = 20,
            sampling_ratio: float = 0.04,
            absolute_sort: bool = True,
        ):
        '''
        if absolute_sort == True: store both the most similar and the most dissimilar users
        '''
        super().__init__()
        assert num_of_users > 0  and top_k > 0
        self._num_of_users = num_of_users
        self._sorting_key = (lambda x: -math.fabs(x[0])) if absolute_sort else (lambda x: -x[0])
        self._overlap_abs_min = 25
        self._overlap_ratio = 0.5
        self._max_p_value = 0.05
        self._sampling_ratio = sampling_ratio
        self._top_k = top_k
        self._top_k_u2u_correlations: List[UserUserWeights.SingleUserTopKSortedEntries] = [
            SortedList(key=self._sorting_key) for _ in range(0, num_of_users)
        ]

    def __setitem__(self, key: int, value: UserUserWeights.SingleUserTopKSortedEntries):
        self._top_k_u2u_correlations[key] = value

    def __getitem__(self, key: int) -> UserUserWeights.SingleUserTopKSortedEntries:
        return self._top_k_u2u_correlations[key]

    def __len__(self) -> int:
        return len(self._top_k_u2u_correlations)

    def get_correlations_as_list(self) -> List[List[CorrelationEntry]]:
        return [[u2u_entry for u2u_entry in user] for user in self._top_k_u2u_correlations]

    def get_correlations(self) -> List[UserUserWeights.SingleUserTopKSortedEntries]:
        return self._top_k_u2u_correlations

    def load_from_matrix(self, centered_rating_matrix: sparse.csr_matrix, shard_id=0, shards_count=1):
        pid = os.getpid()
        print(f'process id: {pid} processing shard {shard_id} out of {shards_count}')
        assert 0 < shard_id + 1 <= shards_count
        # def dummy_pearsonr_algo(x, y):
        #     return DummyPearsonr(statistic=random.uniform(-1.0, 1.0), pvalue=random.uniform(0.3,0.8))
        # pearsonr_algo = dummy_pearsonr_algo
        pearsonr_algo = pearsonr
        def _calculate_corr_ref(user_i:sparse.csr_matrix, user_j:sparse.csr_matrix , common_item_indices: Set[int]):
            '''
            Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
            '''
            user_i_rating_shortlist = np.empty(len(common_item_indices))
            user_j_rating_shortlist = np.empty(len(common_item_indices))
            for out_idx, index_value in enumerate(common_item_indices):
                user_i_rating_shortlist[out_idx] = user_i.data[user_i.indices.searchsorted(index_value)]
                user_j_rating_shortlist[out_idx] = user_j.data[user_j.indices.searchsorted(index_value)]
            return pearsonr_algo(user_i_rating_shortlist, user_j_rating_shortlist)
        def _calculate_corr_array_like(user_i:sparse.csr_matrix, user_j:sparse.csr_matrix , common_item_indices: Set[int]):
            '''
            Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
            '''
            sorted_common_item_indices: List[int] = sorted(common_item_indices)
            user_i_csr_indices = user_i.indices.searchsorted(sorted_common_item_indices)
            user_j_csr_indices = user_j.indices.searchsorted(sorted_common_item_indices)
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
                        self.add_paired_entry((user_i_idx, user_j_idx), corr_value.statistic)
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

    def add_paired_entry(self, two_users: Tuple[int, int], corr_value: int):
        user_i, user_j = two_users
        def _update_user(user: int, other_user: int, value: int) -> UserUserWeights.SingleUserTopKSortedEntries:
            user_top_k_entries: UserUserWeights.SingleUserTopKSortedEntries = self._top_k_u2u_correlations[user]
            user_top_k_entries.add((value, other_user))
            if len(user_top_k_entries) > self._top_k:
                user_top_k_entries.pop()

        _update_user(user_i, user_j, corr_value)
        _update_user(user_j, user_i, corr_value)

    def recreate_entry(self, idx: int, user: SingleUserTopKSortedEntries):
        self._top_k_u2u_correlations[idx] = user

    @property
    def top_k_u2u_correlations(self):
        return self._top_k_u2u_correlations # TODO: convert to dataframe??

    @classmethod
    def serialize(cls, top_k_u2u_correlations: List[SingleUserTopKSortedEntries], filename: str):
        total_written = 0
        u2u_corr_list = [(idx, l) for idx, l in enumerate(top_k_u2u_correlations) if len(l) > 0]
        with open(filename, 'w') as ofile:
            for idx, row in u2u_corr_list:
                output = [str(entry) for entry_tuple in row for entry in entry_tuple]
                ofile.write(f'{ idx}, ' + ', '.join(output) + '\n')
                total_written += 1
                if total_written % LOGGING_PERIOD == 0:
                    print(f"Printed {total_written} lines")

    @staticmethod
    def build_from_files(filenames: List[str]) -> UserUserWeights:
        forced_columns = list([chr(x) for x in range(ord('A'), ord('Z')+1)] +
                              [f'A{chr(x)}' for x in range(ord('A'), ord('Z')+1)])
        u2u_parts = []
        for fname in filenames:
            data = pd.read_csv(
                fname, header=None,
                names=forced_columns,
                skipinitialspace=True,
                escapechar=r"|", doublequote=False # data for legacy experiments
            )
            last_user_idx = int(data.iloc[-1, 0])
            u2u_weights = UserUserWeights(
                num_of_users=max(data.shape[0]+1, last_user_idx+1),
                top_k=20
            )
            # data.shape[1] expected to hold the max len of the longest row

            for index, row in data.iterrows():
                user_id = row.iloc[0].astype(int)
                user_entries = [
                    CorrelationEntry(corr, other_uid)
                    for corr, other_uid
                    in zip(row.iloc[1::2], row.iloc[2::2])
                    if not math.isnan(corr + other_uid)
                ]
                u2u_weights.recreate_entry(user_id, user=SortedList(
                    iterable=user_entries,
                    key=u2u_weights._sorting_key
                ))
                if int(index) % LOGGING_PERIOD == 0:
                    print(f"file: {fname}:{index}")
            u2u_parts.append(u2u_weights)

        u2u_parts = sorted(u2u_parts, key=lambda x: len(x), reverse=True)
        u2u_weights_merged: UserUserWeights = u2u_parts[0]
        for part_idx, part in enumerate(u2u_parts[1:]):
            u2u_weights_merged.merge(part)
            print(f"Merged parts: base + {part_idx + 1}")
        return u2u_weights_merged

    def merge(self, other_u2u_weights: UserUserWeights):
        for other_uid, single_user_row in enumerate(other_u2u_weights.get_correlations()):
            if len(self[other_uid]) == 0:
                self[other_uid] = single_user_row
            else:
                self[other_uid].update(single_user_row)
                pass
            row = self[other_uid]
            while len(row) > self._top_k:
                row.pop()
        return self


def generate_user_user_matrix(pool_size: int = 8) -> None:

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = ratings_matrix_unbiased(build_rating_matrix())
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    sys.stdout.write("Computing user-user correlations...")

    def parallelized_user_user(rmat_centered_with_shard_details):
        rmat_centered, shard_id, shards_count = rmat_centered_with_shard_details
        u2u_w = UserUserWeights(num_of_users=rmat_centered.shape[0], top_k=10)
        u2u_w.load_from_matrix(rmat_centered, shard_id, shards_count)
        return u2u_w.get_correlations_as_list()

    start_time = time.time()
    if pool_size > 1:
        with Pool(pool_size) as p:
            u2u_weights_to_merge = p.map(
                parallelized_user_user,
                [(rating_matrix_centered, i, pool_size) for i in range(0, pool_size)]
            )
    else:
        u2u_weights_to_merge = [parallelized_user_user((rating_matrix_centered, 0, 32))]

    [UserUserWeights.serialize(u2u_w, f"u2u_demo_{idx}.csv") for idx, u2u_w in enumerate(u2u_weights_to_merge)]

    # TODO: configurable option to merge without storing
    print("Done")
    print(f"Time spend on user-user correlations: {time.time() - start_time}")


# head -n 10 u2u_demo_0.csv | sed -E "s/\(([^)]*),([^)]*)\)/'\1|,\2'/g" > u2u_test_inputs_1.csv
def merge_user2user_parts(files: List[str], output: str = "u2u_merged.csv") -> UserUserWeights:
    u2u_merged = UserUserWeights.build_from_files(files)
    print(len(u2u_merged))
    all_lengths = [(idx, len(entry)) for idx, entry in enumerate(u2u_merged.get_correlations_as_list())]
    print(len(['' for idx, entry_len in all_lengths if entry_len > 0]))
    UserUserWeights.serialize(u2u_merged.get_correlations(), "u2u_merged.csv")
    return u2u_merged


class Recommender:
    def __init__(self, ratings_matrix_unbiased: sparse.csr_matrix, u2u_weights: UserUserWeights):
        self._rating_mat = rating_matrix_centered
        self._u2u_weights = u2u_weights

    def recommend(self, user_id: int) -> List[int]:
        correlated_users = self._u2u_weights[user_id]
        omega_NB_movie_ids: Set[int] = set() # movies watched by all neighbors
        omega_NB_movies: List[int] = []
        rows_tmp = []
        for corr, neighbor_id in correlated_users:
            neighbor_id = int(neighbor_id)
            neighbor_movies = self._rating_mat.getrow(neighbor_id)
            rows_tmp.append(neighbor_movies)
            omega_NB_movies.append(neighbor_movies.indices.tolist())
            omega_NB_movie_ids = omega_NB_movie_ids.union(neighbor_movies.indices.tolist())

        print(f"Number of movies watched by neighbors {len(omega_NB_movie_ids)}")
        import pdb; pdb.set_trace()
        # TODO

        return []


if __name__ == '__main__':

    # u2u_merged = merge_user2user_parts([
    #     f"u2u_test_inputs_{idx}.csv" for idx in range(0, 8)
    # ])
    # import pdb; pdb.set_trace()
    u2u_merged = UserUserWeights.build_from_files(["u2u_merged.csv"])

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = ratings_matrix_unbiased(build_rating_matrix())
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    rec = Recommender(
        ratings_matrix_unbiased=rating_matrix_centered,
        u2u_weights=u2u_merged
    )

    rec.recommend(user_id=99623)
    import pdb; pdb.set_trace()