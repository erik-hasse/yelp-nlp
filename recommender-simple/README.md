# Providing restaurant recommendations

## Running instructions

1. Create an environment using `conda.yaml`. There is also a requirements.txt file, but using it alone may lead to version incompatibilities. Make sure its active for the remainder of these steps.
2. Download the [Yelp dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset) and extract it to the `data` folder in the repo root.
3. From the `recommender` directory, run `python src/train_retrieval.py`. This will perform all data preparation and training to create a retrieval model.
4. To serve the (mock) api, run `uvicorn src.recommender_app:app` from the `price-predictions` directry.

## Overview

The goal is to generate recommendations to a user based on their profile and city.
My goal was to do this in a two step process.
First, narrow down the restaurants using embeddings to find restaurants that are rated by similar people - this is the retrieval model.
Then second, take the output of the retrieval model and score each restaurant by how likely the user is to enjoy it - this is the ranking model.

## Things tried

Due to time constraints, I was unfortunately unable to complete a model for this task.
My retrieval model runs, however due to the large number of reviewers with just a single review, the model very often returns just a single restaurant repeated multiple times.

I was unable to build a ranking model, but I do have some ideas about how to go about it.
Asa starting point, my goal would be to use both a sentiment analysis of the review text and the star rating they gave the restaurant.
This provides a two sources of ratings, which should be complimentary.
The sentiment analysis could be provided by 🤗 Transformers `pipeline(sentiment-analysis)` in the initial case, although it only provides "positive" or "negative".
This approach does still have a problem with the city requirement however.
In the user's home city, they are likely similar enough to other users that the retrieval model would return enough results.
However, in an unknown city, the retrival model may not return anything in the new city.
This may be solvable by adding an input to the retrieval model so that it first filters to the given city.

As enhancements, I would want to use a more complex sentiment analysis model, potentially even determining specific positive and negative elements (e.g. "good food, but too expensive").
In addition, the attributes of a restarant are certainly useful if they can be mapped to the user's existing ratings.
Finally, the restaurants enjoyed by a user's friends would be a great starting point for the recommendations.
These could potentially be automatically included in the input to the ranking model.
