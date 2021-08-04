from itertools import islice
import json

import tensorflow as tf

from src.constants import data_dir


def check_data_file(name):
    file = data_dir / name
    if not file.is_file():
        raise RuntimeError(
            f'Could not find {name}. Please download it and place it in '
            'the data directory.'
        )


def load_data(n_examples=None):
    file_name = 'yelp_academic_dataset_review.json'
    check_data_file(file_name)
    data = {
        'user_id': [],
        'business_id': [],
        'stars': [],
        'text': [],
        'date': []
    }
    with open(data_dir / file_name, 'rb') as f:
        for x in islice(f, n_examples):
            review = json.loads(x)
            data['user_id'].append(review['user_id'])
            data['business_id'].append(review['business_id'])
            data['stars'].append(int(review['stars']))
            data['text'].append(review['text'].replace('\n', ''))
            data['date'].append(review['date'])

    return tf.data.Dataset.from_tensor_slices(data)


def get_unique_ids(dataset):
    file_name = f'yelp_academic_dataset_{dataset}.json'
    check_data_file(file_name)
    all_ids = set()
    with open(data_dir / file_name, 'rb') as f:
        for x in f:
            item = json.loads(x)
            all_ids.add(item[f'{dataset}_id'])

    return list(all_ids)
