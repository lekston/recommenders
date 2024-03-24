from __future__ import annotations

import math

from scipy import sparse
from typing import Dict, List, Set, Tuple, TypeAlias

from .cf_weights import (
    CollaborativeFilteringWeights
)

class BasicUser2UserRecommender:
    def __init__(self, ratings_matrix_unbiased: sparse.csr_matrix, u2u_weights: CollaborativeFilteringWeights):
        self._rating_mat = ratings_matrix_unbiased
        self._u2u_weights = u2u_weights

    def recommend(self, user_id: int, min_reviews_per_movie: int = 5) -> List[Tuple[int, float]]:
        '''
        TODO: potential optimizations:
        - cache recommendation vector for each requested user
        '''
        print(f"Processing user {user_id}")
        correlated_users = self._u2u_weights[user_id]
        if len(correlated_users) < 3:
            raise RuntimeError(f"Recommendation Error:"
                               f" User {user_id} has too few {len(correlated_users)} correlated users")

        MovieRatingNumDenum: TypeAlias = Tuple[float, float, int]
        movie_scores_wip: Dict[int, MovieRatingNumDenum] = {}
        omega_NB_movie_ids: Set[int] = set() # movies watched by all neighbors

        def _update_movie_scores(movie_id: int, movie_rating: float, u2u_weight: float):
            rating_num_denum_reviewers = (u2u_weight * movie_rating, math.fabs(u2u_weight), 1)
            if movie_id in movie_scores_wip.keys():
                prev_score = movie_scores_wip[movie_id]
                movie_scores_wip[movie_id] = (
                    prev_score[0] + rating_num_denum_reviewers[0],
                    prev_score[1] + rating_num_denum_reviewers[1],
                    prev_score[2] + rating_num_denum_reviewers[2]
                )
            else:
                movie_scores_wip[movie_id] = rating_num_denum_reviewers

        for corr_u2u_weight, neighbor_id in correlated_users:
            neighbor_id = int(neighbor_id)
            neighbor_movies = self._rating_mat.getrow(neighbor_id)
            omega_NB_movie_ids = omega_NB_movie_ids.union(neighbor_movies.indices.tolist())

            for movie_id, movie_rating in zip(neighbor_movies.indices, neighbor_movies.data):
                _update_movie_scores(movie_id, movie_rating, corr_u2u_weight)

        print(f"Number of movies watched by neighbors {len(omega_NB_movie_ids)} ({len(movie_scores_wip.keys())})")
        movie_scores: List[Tuple[int, float]] = sorted(
            [
                (movie_id, score_num/score_denum)
                for movie_id, (score_num, score_denum, reviewers) in movie_scores_wip.items()
                if reviewers >= min_reviews_per_movie
            ],
            key=lambda movie_id_score: movie_id_score[1],
            reverse=True
        )
        print(f"Total recommendations: {len(movie_scores)}."
              f" Best (incl. already watched) {movie_scores[:10]}")
        already_watched_indices = self._rating_mat[user_id].indices
        unwatched_movie_scores = list([
            (movie_id, score) for movie_id, score in movie_scores if movie_id not in already_watched_indices
        ])
        print(f"Unwatched recommendations: {len(unwatched_movie_scores)}."
              f" Best (unwatched): {unwatched_movie_scores[:10]}")
        print(f" Worst (unwatched): {unwatched_movie_scores[-10:]}")
        print("Removing already watched")
        already_watched_with_scores = [
            (mv_id, score) for mv_id, score
            in zip(self._rating_mat[user_id].indices, self._rating_mat[user_id].data)
        ]

        print("Comparing scores of already watched vs. preditions")
        predicted_watched_movie_scores = list([
            (movie_id, score) for movie_id, score in movie_scores if movie_id in already_watched_indices
        ])
        overlapped_indices = [movie_id for movie_id, score in predicted_watched_movie_scores]
        overlapped_ground_truth: List[Tuple[int, float]] = sorted(
            [
                (movie_id, score)
                for movie_id, score in already_watched_with_scores
                if movie_id in overlapped_indices
            ],
            key=lambda movie_id_score: movie_id_score[1],
            reverse=True
        )

        for prediction, gt in zip(predicted_watched_movie_scores, overlapped_ground_truth):
            print(f"prediction: {prediction}, gt: {gt}")

        print(f"Recommendations for user {user_id}:")
        [print(item) for item in unwatched_movie_scores[:20]]
        return unwatched_movie_scores
