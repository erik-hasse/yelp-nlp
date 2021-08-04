import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.prep_data import load_data


def train():
    orig_data = load_data(400_000)
    data = orig_data.map(lambda x: {
        "business_id": x["business_id"],
        "user_id": x["user_id"],
    })

    businesses = orig_data.map(lambda x: x['business_id'])
    users = orig_data.map(lambda x: x['user_id'])
    unique_businesses = np.unique([v.numpy() for v in businesses])
    unique_user_ids = np.unique([v.numpy() for v in users])

    tf.random.set_seed(42)
    shuffled = data.shuffle(400_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(320_000)
    test = shuffled.skip(320_000).take(80_000)

    embedding_dimension = 32

    user_model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None
        ),
        tf.keras.layers.Embedding(
            len(unique_user_ids) + 1, embedding_dimension
        )
    ])

    business_model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_businesses, mask_token=None
        ),
        tf.keras.layers.Embedding(
            len(unique_businesses) + 1, embedding_dimension
        )
    ])

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=businesses.batch(128).map(business_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )

    class YelpModel(tfrs.Model):
        def __init__(self, user_model, movie_model):
            super().__init__()
            self.business_model: tf.keras.Model = business_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features, training=False) -> tf.Tensor:
            user_embeddings = self.user_model(features['user_id'])
            business_embeddings = self.business_model(features['business_id'])

            return self.task(user_embeddings, business_embeddings)

    model = YelpModel(user_model, business_model)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(320_000).batch(8000).cache()

    model.fit(cached_train, epochs=1)

    return model, test


def evaluate(model, test):
    cached_test = test.batch(4000).cache()
    model.evaluate(cached_test, return_dict=True)


def save_model(model):
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.save('results')


if __name__ == '__main__':
    model, test_data = train()
    evaluate(model, test_data)
    save_model(model)
