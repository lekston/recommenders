from __future__ import annotations

import sys
import time

from src.user_user_weights import (
    UserUserWeights
)
from src.basic_recommender import BasicRecommender
from src.utils import (
    build_rating_matrix,
    unbias_the_ratings,
    generate_user_user_matrix,
    merge_user2user_parts
)

'''
NOTE: All input sparse matrices / arrays use indexing that starts with 1 (userId & movieId)
work-around:
  userId = 0 is a dummy user
  movieId = 0 is a dummy movie
'''

def demo_recommend() -> None:
    # u2u_merged = UserUserWeights.build_from_files(["./workdir/u2u_merged.csv"])
    u2u_merged = UserUserWeights.build_from_files(["./u2u_merged.csv"])

    sys.stdout.write("Centering user ratings...")
    start_time = time.time()
    rating_matrix_centered = unbias_the_ratings(build_rating_matrix())
    print("Done")
    print(f"Time spend on bias removal: {time.time() - start_time}")

    rec = BasicRecommender(
        ratings_matrix_unbiased=rating_matrix_centered,
        u2u_weights=u2u_merged
    )

    #rec.recommend(user_id=99623)
    #rec.recommend(user_id=5024)
    rec.recommend(user_id=70000)
    rec.recommend(user_id=11111)


def demo_preprocess() -> None:
    generate_user_user_matrix(pool_size=8, merged_output_file='u2u_merged.csv')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'preprocess':
        print("Note: the demo config of this run takes ~5 hours on 4 core i7.")
        answer = input("Do you want to proceed? [y/n]")
        if answer != 'y':
            print("Aborting")
            sys.exit(0)
        demo_preprocess()
    elif len(sys.argv) == 2 and sys.argv[1] == 'demo_recommend':
        demo_recommend()
    else:
        print("Usage:")
        print(f"\t{sys.argv[0]} preprocess - runs User-User weight matrix generation")
        print(f"\t{sys.argv[0]} demo_recommend - runs recommendation example")


