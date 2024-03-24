from __future__ import annotations

import math
import numpy as np
import os
import pandas as pd
import time

from sortedcollections import SortedList

from scipy import sparse
from scipy.stats import pearsonr
from scipy.stats._result_classes import PearsonRResult
from typing import List, NamedTuple, Set, Tuple, TypeAlias


class CorrelationEntry(NamedTuple):
    correlation: float
    neighbor_id: int


class DummyPearsonr(NamedTuple):
    '''
    Dummy value for use in performance profiling
    '''
    statistic: float
    pvalue: float


LOGGING_PERIOD = 20000


class CollaborativeFilteringWeights(object):
    TopKSortedNeighbors : TypeAlias = SortedList[CorrelationEntry]
    '''
    This class can be used to calculate correlations for both User-User and Item-Item CF.

    However, some naming conventions have been adopted to simplify understanding of the
    default User-User case.

    Most method internals assume that we work with User-User correlations. This can be generalized by assuming
    that `user` corresponds to `row` while `idem` to column.
    For Item-Item correlations case the meaning of `user` and `item` has to be swapped (as does `u2u` & `i2i`).
    '''

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
        self._sampling_ratio = sampling_ratio if 0 < sampling_ratio < 1.0 else 1
        self._top_k = top_k
        self._top_k_u2u_correlations: List[CollaborativeFilteringWeights.TopKSortedNeighbors] = [
            SortedList(key=self._sorting_key) for _ in range(0, num_of_users)
        ]
        self._pearsonr_algo = pearsonr
        # def dummy_pearsonr_algo(x, y):
        #     return DummyPearsonr(statistic=random.uniform(-1.0, 1.0), pvalue=random.uniform(0.3,0.8))
        # self._pearsonr_algo = dummy_pearsonr_algo
        

    def __setitem__(self, key: int, value: CollaborativeFilteringWeights.TopKSortedNeighbors):
        self._top_k_u2u_correlations[key] = value

    def __getitem__(self, key: int) -> CollaborativeFilteringWeights.TopKSortedNeighbors:
        return self._top_k_u2u_correlations[key]

    def __len__(self) -> int:
        return len(self._top_k_u2u_correlations)

    def get_correlations_as_list(self) -> List[List[CorrelationEntry]]:
        return [list(user) for user in self._top_k_u2u_correlations]

    def get_correlations(self) -> List[CollaborativeFilteringWeights.TopKSortedNeighbors]:
        return self._top_k_u2u_correlations

    def _calculate_corr_(
            self,
            user_i:sparse.csr_matrix,
            user_j:sparse.csr_matrix,
            common_item_indices: Set[int]
        ):
        '''
        Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
        '''
        user_i_rating_shortlist = np.empty(len(common_item_indices))
        user_j_rating_shortlist = np.empty(len(common_item_indices))
        for out_idx, index_value in enumerate(common_item_indices):
            user_i_rating_shortlist[out_idx] = user_i.data[user_i.indices.searchsorted(index_value)]
            user_j_rating_shortlist[out_idx] = user_j.data[user_j.indices.searchsorted(index_value)]
        return self._pearsonr_algo(user_i_rating_shortlist, user_j_rating_shortlist)

    def _calculate_corr_array_like(
            self,
            user_i:sparse.csr_matrix,
            user_j:sparse.csr_matrix,
            common_item_indices: Set[int]
        ):
        '''
        Extract shared ratings i.e. for items that both users have rated and compute user-user correlation
        '''
        sorted_common_item_indices: List[int] = sorted(common_item_indices)
        user_i_csr_indices = user_i.indices.searchsorted(sorted_common_item_indices)
        user_j_csr_indices = user_j.indices.searchsorted(sorted_common_item_indices)
        return self._pearsonr_algo(user_i.data[user_i_csr_indices], user_j.data[user_j_csr_indices])


    def load_from_matrix(self, centered_rating_matrix: sparse.csr_matrix, shard_id=0, shards_count=1):
        pid = os.getpid()
        print(f'process id: {pid} processing shard {shard_id} out of {shards_count}')
        assert 0 < shard_id + 1 <= shards_count

        start_time = time.time()
        num_updated = 0
        for user_i_idx in range(1, centered_rating_matrix.shape[0]):
            user_i = centered_rating_matrix.getrow(user_i_idx)
            lower_j_idx = 1 + int((shard_id)/shards_count * user_i_idx)
            upper_j_idx = int((shard_id+1)/shards_count * user_i_idx)
            size_of_subset = int(self._sampling_ratio*(upper_j_idx - lower_j_idx))
            if self._sampling_ratio < 1:
                # poor-man's Monte-Carlo selection
                selection_subset = np.unique(np.random.uniform(lower_j_idx, upper_j_idx, size_of_subset).astype(int))
            else:
                selection_subset = range(lower_j_idx, upper_j_idx)
            for user_j_idx in selection_subset:
                user_j = centered_rating_matrix.getrow(user_j_idx)
                common_item_indices = set(user_i.indices).intersection(set(user_j.indices))
                required_overlap = min(len(user_i.indices), len(user_j.indices)) * self._overlap_ratio
                required_overlap = max(required_overlap, self._overlap_abs_min)
                if len(common_item_indices) > required_overlap:
                    # process users with high overlap
                    # corr_value: PearsonRResult = self._calculate_corr_ref(user_i, user_j, common_item_indices)
                    corr_value: PearsonRResult = self._calculate_corr_array_like(user_i, user_j, common_item_indices)
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
        def _update_user(user: int, other_user: int, value: int) -> CollaborativeFilteringWeights.TopKSortedNeighbors:
            user_top_k_entries: CollaborativeFilteringWeights.TopKSortedNeighbors = self._top_k_u2u_correlations[user]
            user_top_k_entries.add((value, other_user))
            if len(user_top_k_entries) > self._top_k:
                user_top_k_entries.pop()

        _update_user(user_i, user_j, corr_value)
        _update_user(user_j, user_i, corr_value)

    def recreate_entry(self, idx: int, user: TopKSortedNeighbors):
        self._top_k_u2u_correlations[idx] = user

    @property
    def top_k_u2u_correlations(self):
        return self._top_k_u2u_correlations # TODO: convert to dataframe??

    @classmethod
    def serialize(cls, list_of_top_k_correlations: List[TopKSortedNeighbors], filename: str):
        total_written = 0
        u2u_corr_list = [(idx, l) for idx, l in enumerate(list_of_top_k_correlations) if len(l) > 0]
        with open(filename, 'w') as ofile:
            for idx, row in u2u_corr_list:
                output = [str(entry) for entry_tuple in row for entry in entry_tuple]
                ofile.write(f'{ idx}, ' + ', '.join(output) + '\n')
                total_written += 1
                if total_written % LOGGING_PERIOD == 0:
                    print(f"Printed {total_written} lines")

    @staticmethod
    def build_from_files(filenames: List[str]) -> CollaborativeFilteringWeights:
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
            u2u_weights = CollaborativeFilteringWeights(
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
        u2u_weights_merged: CollaborativeFilteringWeights = u2u_parts[0]
        for part_idx, part in enumerate(u2u_parts[1:]):
            u2u_weights_merged.merge(part)
            print(f"Merged parts: base + {part_idx + 1}")
        return u2u_weights_merged

    def merge(self, other_cf_weights: CollaborativeFilteringWeights):
        for other_uid, single_user_row in enumerate(other_cf_weights.get_correlations()):
            if len(self[other_uid]) == 0:
                self[other_uid] = single_user_row
            else:
                self[other_uid].update(single_user_row)
                pass
            row = self[other_uid]
            while len(row) > self._top_k:
                row.pop()
        return self

