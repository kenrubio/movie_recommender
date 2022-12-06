docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

docker build . -t movie_recommender
docker run movie_recommender --model="mf" --movie_title="Back to the Future" --num_reco=5