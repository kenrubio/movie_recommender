# Movie Recommendation System

A KNN-based and an Item-based Collaborative Filtering recommender systems that recommend movies based on the movie title provided.

## Overview

The movies are recommended based on the title of the movie provided. The main feature that is considered for the recommendations the movie ratings provided by the users. The details of the movies, such as title, movieId, userId, and rating are fetched from [MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) data set.


## How to run the project?

1. Open the run.sh file.
2. Change the parameters of the docker run command:
    - model: knn (K-Nearest Neighbor), mf (item-based matrix factorization)
    - movie_title: Any name of the movie you like
    - num_reco: Number of recommendations to produce
3. Save the changes.
4. Open your terminal/command prompt from your project directory and run the file `run.sh` by executing the command `sh run.sh`.
5. That's it.
