# Predicing price from reviews

## Running instructions

1. Create an environment using `conda.yaml`. There is also a requirements.txt file, but using it alone may lead to version incompatibilities. Make sure its active for the remainder of these steps.
2. Download the [Yelp dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset) and extract it to the `data` folder in the repo root.
3. From the `price-predictions` directory, run `python src/train.py`. This will perform all data preparation and training, and save a `conf-matrix.png` file for reviewing results on a test set.
4. To serve the api, run `uvicorn src.review_app:app` from the `price-predictions` directry.

As an alternate method to only serve the API, I have prepared a docker image.
After downloading it [here](https://1drv.ms/u/s!AnQnLRYo5rEWlMItdaf-xqLYVjwL_w?e=YGycwV) load it with `docker load < price-predictions.tar`, then run `docker run -it -p 8000:8000 price-predictions`.

## Overview

The goal is to predict the price of a restaurant from its reviews.
This is ultimately a tough task, because price rarely mentioned directly in reviews, and so the text is at best weakly predictive.
However, it provides a useful framework to understand the structure of the Yelp dataset, as well as an introduction to the 🤗 Transformers library.

## Things tried

Due to computational and time constraints, I have only tried two simple models.
In both cases I used the `TFDistilBertForSequenceClassification` pre-trained model from 🤗 Transformers, and further trained it for the task on the Yelp data.
The difference between the two runs that I tried was consering the problem as regression versus classification.
There are reasonable arguments to be made for either - PriceRange is a numeric feature so regression does make sense, but its range may not be evenly spaced, making it hard to directly interpret.
Further, due to computational limits I restricted the amount of data used by the model.
In any case the results were the same, the model always predicts approximately the mean of the training set.
As discussed above, this is not a terribly surprising result, because reviews are at best weakly predictive of price.

![](./conf-matrix.png?raw=true)
