import pandas as pd
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def load_csv(file_path: str):
    """Loads the data using the file path.

    Args:
        file_path (str): file location

    Returns:
        dataframe: loaded data into a dataframe
    """

    input_data = pd.read_csv(file_path)

    return input_data


def create_dummies(df, col, sep):
    """Converts a column of entries with multiple categories
    into a one-hot encoding or indicator variables.

    Args:
        df (dataframe): input data
        col (str): column name to convert into an encoding
        sep (str): separator character used in each entry

    Returns:
        res_df (dataframe): data with converted column into
                            one-hot encoding
    """
    res_df = df[col].str.get_dummies(sep=sep)

    return res_df


def drop_cols(df, col):
    """Drop columns from the dataframe.

    Args:
        df (dataframe): input data
        col (str): column to be dropped
    """
    df.drop(columns=col, axis=1, inplace=True)

    return df


def get_movie_id(df, movie_title):
    """Provides the movie id using the given movie title

    Args:
        df (dataframe): dataframe of movie names and ratings
        movie_title (str): name of the movie

    Returns:
        int: id of the movie title
    """
    movie_title, rel_names = get_titles(df, movie_title)
    movie_id = df[df["title"] == movie_title]["movieId"].values

    if len(movie_id) > 0:
        movie_id = movie_id[0]

        if len(rel_names) > 0:
            print(f"Here are the other related titles to your input: {', '.join(rel_names)} \n")

        else:
            print("There are no similarly named movies in the database...")
        
        print(f'Because you watched {movie_title}:')

    else:
        movie_id = None

    return movie_id


def get_titles(df, movie_title):
    """Get the top title and other related titles to the input movie name

    Args:
        df (dataframe): dataframe of movie names and ratings
        movie_title (str): name of the movie of interest

    Returns:
        str: name of top title of the input
        list: other related titles of the input
    """
    movie_titles = df[df['title'].str.contains(movie_title)]["title"].unique()

    if len(movie_titles) != 0:
        dictio = {}

        for name in movie_titles:
            dictio[name] = fuzz.ratio(movie_title, name)

        # Rank the similar movie titles
        # and arrange in descending order
        name_rank = pd.DataFrame.from_dict(dictio,
                                           orient='index') \
                                .reset_index()
        name_rank = name_rank.rename(columns={'index': 'title',
                                              0: 'score'}) \
                             .sort_values(by="score",
                                          ascending=False)

        top_title = name_rank.iloc[0]["title"]
        other_titles = name_rank.iloc[1:]["title"].tolist()

        return top_title, other_titles

    else:
        return None, None


def preprocess_metadata():
    """Prepares data sets to create user-item interaction

    Args:
        None

    Returns:
        list: collection of movie and rating dataframes
    """
    # Load data sets
    df_movie = load_csv("project/data/movie.csv")
    df_rating = load_csv("project/data/rating.csv")

    # Get the number of rating given by each user
    vote_counts = pd.DataFrame(df_rating["userId"]
                               .value_counts()).reset_index()
    vote_counts.columns = ["user", "vote_count"]

    # Take the 90% quantile since we want movies which
    # have got votes done more than 90% of the total
    minimum_vote_count = vote_counts["vote_count"].quantile(0.9)

    # Isolate users with votes above the 90th quantile
    active_voters = vote_counts[vote_counts["vote_count"] >=
                                minimum_vote_count]["user"]

    # Only include users who actively rates/votes
    df_rating = df_rating[df_rating["userId"].isin(active_voters)]

    # Combine the ratings and movie data
    movie_metadata = pd.merge(df_movie, df_rating, on="movieId")

    # Remove unnecessary feature
    movie_metadata.drop(columns=['timestamp'], inplace=True)

    # Get the number of rating given per movie
    rating_counts = movie_metadata.groupby('title')['rating'] \
                                  .count() \
                                  .reset_index()
    rating_counts.rename(columns={"rating": "rating_count"},
                         inplace=True)

    # Combine the rating counts to the movie metadata
    df = movie_metadata.merge(rating_counts, on='title')

    # Take the 90% quantile since we want movies
    # which have got ratings more than 90% of the total
    minimum_rating_count = df["rating_count"].quantile(0.9)

    # Only include movies that are rated more
    # than 90% of the total rating count
    df = df[df["rating_count"] >= minimum_rating_count]

    # Remove duplicate rows
    df.drop_duplicates(['title', 'userId'], inplace=True)

    # Remove no longer useful feature
    df.drop(columns=['rating_count'], inplace=True)

    # Create a pivot table for the user-movie interaction
    df_user_movie = df.pivot_table(index="movieId",
                                   columns="userId",
                                   values="rating")
    df_user_movie = df_user_movie.fillna(0)

    # Remove no longer useful features
    df.drop(columns=['genres', 'userId', 'rating'],
            inplace=True)

    # Remove duplicates after dropping features
    df.drop_duplicates(inplace=True)

    # Use csr_matrix to reduce the sparsity of the data
    sparse_df = csr_matrix(df_user_movie)

    return df, df_user_movie, sparse_df


def create_knn_model(df, num_reco):
    """Instantiates a model with brute algorithm and cosine similarity metric

    Args:
        df (dataframe): dataframe of user-item interaction
        num_reco (int): number of recommendations to produce

    Returns:
        object: knn model
    """
    # Instantiate model using cosine similarity
    model = NearestNeighbors(n_neighbors=num_reco+1,
                             algorithm='brute',
                             metric='cosine')
    model.fit(df)

    return model


def get_recommendations_knn(movie_title, base_df, user_movie_df, model):
    """Provides movie names that are close neighbors of the input name

    Args:
        movie_title (str): name of the movie of interest
        base_df (dataframe): base movie-rating metadata
        user_movie_df (dataframe): user-item interaction dataframe
        model (object): knn model

    Returns:
        list: names of recommended movies
    """
    movie_title, rel_names = get_titles(base_df, movie_title)

    if movie_title is not None:
        movie_id = base_df[base_df['title'] == movie_title] \
                        .drop_duplicates('title')['movieId'] \
                        .values[0]
        user_ratings = user_movie_df[user_movie_df.index == movie_id] \
            .values \
            .reshape(1, -1)

        distances, suggestions = model.kneighbors(user_ratings)
        suggestions = (suggestions.flatten()[1:]).tolist()
        distances = distances.flatten()[1:]

        recommendations = [base_df[base_df["movieId"] ==
                                   user_movie_df.index[i]]["title"]
                           .drop_duplicates()
                           .values[0]
                           for i in suggestions]

        if len(rel_names) > 0:
            print(f"Here are the other related titles to your input: {', '.join(rel_names)} \n")

        else:
            print("There are no similarly named movies in the database...")

        print(f'Because you watched {movie_title}:')

        for i in range(len(recommendations)):
            print(f'{i+1}: {recommendations[i]}')

        return recommendations

    else:
        print("We're sorry. Your movie is not available in our database...")
        return None


def get_recommendations_mf(movie_id, num_reco, base_df, df_user_movie):
    """Provides movie names using item-based collaborative filtering

    Args:
        movie_id (int): id of the movie of interest
        num_reco (int): number of recommendations to produce
        base_df (dataframe): base movie-rating metadata
        df_user_movie (dataframe): dataframe of user-item interaction

    Returns:
        list: names of recommended movies
    """

    if movie_id is not None:
        # Isolate the rating values of each user for the movie
        df_movie_rating = df_user_movie[df_user_movie.index == movie_id]

        # Squeeze the dataframe to create a Series
        series_movie_rating = df_movie_rating.squeeze()

        # Transpose the dataframe to calculate the correlation values
        movie_rank = df_user_movie.T \
            .corrwith(series_movie_rating) \
            .sort_values(ascending=False)

        recommendations = movie_rank.iloc[1:num_reco+1].index.tolist()

        for i in range(len(recommendations)):
            movie_title = base_df[base_df["movieId"] ==
                                 recommendations[i]]["title"].values[0]
            print(f'{i+1}: {movie_title}')

        return recommendations

    else:
        print("We're sorry. Your movie is not available in our database...")
        return None


if __name__ == "__main__":
    input_data = load_csv("../data/rating.csv")
    print(input_data.head())
