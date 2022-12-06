import argparse

from project.utils.dataframe_processor import (create_knn_model, get_movie_id,
                                               get_recommendations_knn,
                                               get_recommendations_mf,
                                               preprocess_metadata)

parser = argparse.ArgumentParser(
    description='Recommender system for movies/films')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    help="Model type: knn or mf")
parser.add_argument('-mt',
                    '--movie_title',
                    type=str,
                    help="Name of the movie")
parser.add_argument('-nr',
                    '--num_reco',
                    type=int,
                    help="Number of recommendations")
args = vars(parser.parse_args())


if __name__ == "__main__":
    if args["movie_title"] is not None:
        movie_title = args["movie_title"]

    if args["num_reco"] is not None:
        num_reco = args["num_reco"]

    base_df, df_user_movie, sparse_df = preprocess_metadata()

    if args["model"] == "mf":
        movie_id = get_movie_id(base_df, movie_title)
        recommendations = get_recommendations_mf(movie_id,
                                                num_reco,
                                                base_df,
                                                df_user_movie)

    elif args["model"] == "knn":
        model = create_knn_model(sparse_df, num_reco)
        recommendations = get_recommendations_knn(movie_title,
                                                base_df,
                                                df_user_movie,
                                                model)
