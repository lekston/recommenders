from __future__ import annotations

import sys
import time

from src.cf_weights import (
    CollaborativeFilteringWeights
)
from src.basic_recommender import BasicUser2UserRecommender
from src.utils import (
    build_rating_matrix,
    calc_collaborative_filtering_weights,
    CFilterType,
    unbias_the_ratings,
)

'''
NOTE: All input sparse matrices / arrays use indexing that starts with 1 (userId & movieId)
work-around:
  userId = 0 is a dummy user
  movieId = 0 is a dummy movie
'''

def demo_recommend_u2u() -> None:
    # u2u_merged = CollaborativeFilteringWeights.build_from_files(["./workdir/u2u_merged.csv"])
    u2u_merged = CollaborativeFilteringWeights.build_from_files(["./u2u_merged.csv"])

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = unbias_the_ratings(build_rating_matrix())
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    rec = BasicUser2UserRecommender(
        ratings_matrix_unbiased=rating_matrix_centered,
        u2u_weights=u2u_merged
    )

    #rec.recommend(user_id=99623)
    #rec.recommend(user_id=5024)
    rec.recommend(user_id=70000)
    rec.recommend(user_id=11111)


def demo_recommend_i2i() -> None:
    i2i_merged = CollaborativeFilteringWeights.build_from_files(["./i2i_merged.csv"])

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = unbias_the_ratings(build_rating_matrix())
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    # TODO Implement BasicItem2ItemRecommender
    '''
    Compute predicted score:
        - get all items j \subset Psi_u(i) rated by user i
        - select all items NB_i(j) which are correlated with items j \subset Psi_u(i)
        - calculate scores for all items (NB_i(j) \setminus Psi_u(i)) not seen by user i
            - each item from NB_i(j) may be similar / dissimilar to several movies rated by the user i
            - score for items from NB_i(j) is calculated by weighted sum over all similar/dissimilar items j' \subset Psi_u(i)
            - weights come from the item-item weights
            - deviations come from the ratings of the current user only
        - recommend the best match
    '''
    pass


def demo_preprocess_u2u() -> None:
    calc_collaborative_filtering_weights(
        pool_size=8,
        merged_output_file='u2u_merged.csv',
        run_type=CFilterType.user2user
    )


def demo_preprocess_i2i() -> None:
    calc_collaborative_filtering_weights(
        pool_size=8,
        merged_output_file='i2i_merged.csv',
        run_type=CFilterType.item2item
    )



if __name__ == '__main__':
    def _print_usage():
        print("Usage:")
        print(f"\t{sys.argv[0]} u2u_preprocess - runs User-User weight matrix generation")
        print(f"\t{sys.argv[0]} u2u_demo_recommend - runs User-User recommendation example")
        print(f"\t{sys.argv[0]} i2i_preprocess - runs Item-Item weight matrix generation")
        print(f"\t{sys.argv[0]} i2i_demo_recommend - runs Item-Item recommendation example")

    if len(sys.argv) == 2:
        if sys.argv[1] == 'u2u_preprocess':
            print("Note: the demo config of this run takes ~5 hours on 4 core i7.")
            answer = input("Do you want to proceed? [y/n]")
            if answer != 'y':
                print("Aborting")
                sys.exit(0)
            demo_preprocess_u2u()
        elif sys.argv[1] == 'u2u_demo_recommend':
            demo_recommend_u2u()
        elif sys.argv[1] == 'i2i_preprocess':
            print("Note: the demo config of this run takes long time to run.")
            answer = input("Do you want to proceed? [y/n]")
            if answer != 'y':
                print("Aborting")
                sys.exit(0)
            demo_preprocess_i2i()
        elif sys.argv[1] == 'i2i_demo_recommend':
            demo_recommend_i2i()
        else:
            _print_usage()
    else:
        _print_usage()




