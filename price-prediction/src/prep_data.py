import json

from constants import data_dir


def get_price(data):
    """Decode a string or bytes representation of a business and return a tuple
    of business_id, price.

    """
    data = json.loads(data)
    bus_id = data.get('business_id')
    price = (data.get('attributes') or {}).get('RestaurantsPriceRange2')
    try:
        price = int(price)
    except (ValueError, TypeError):
        price = None
    return bus_id, price


def check_data_file(name):
    file = data_dir / name
    if not file.is_file():
        raise RuntimeError(
            f'Could not find {name}. Please download it and place it in '
            'the data directory.'
        )


def get_business_price_data():
    """Return a dictionary of business prices. The keys are business ids and
    the values are the prices (or None if a prices is not available).

    """
    file_name = 'yelp_academic_dataset_business.json'
    check_data_file(file_name)
    with open(data_dir / file_name, 'rb') as file:
        return dict(get_price(line) for line in file)


def get_review_data():
    """Return a tuple of (review texts, business prices). Only reviews with a
    price will be returned.

    """
    business_data = get_business_price_data()

    def clean_review(data):
        """Decode a string or bytes representation of a review and return a
        tuple of review text and price. Newlines will be removed from the
        review.

        """
        data = json.loads(data)
        return (
            data['text'].replace('\n', ''),
            business_data.get(data['business_id'])
        )

    file_name = 'yelp_academic_dataset_review.json'
    check_data_file(file_name)

    with open(data_dir / file_name, 'rb') as file:
        # Loop over the lines, pass each one to clean_review, and drop the
        # ones with None for price
        reviews_with_prices = [
            rev_data for line in file
            if (rev_data := clean_review(line))[1] is not None
        ]

    # Transpose the data to return a tuple of texts and a tuple of prices
    return tuple(zip(*reviews_with_prices))


def split_data(train_size=0.6, validate_size=0.2, test_size=0.2):
    """Split the dataset into train, test, and validate sets and save
    them to individual files. Proportions can be set with arguments.

    """
    from sklearn.model_selection import train_test_split

    reviews, prices = get_review_data()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        reviews, prices, test_size=test_size, random_state=42
    )
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_val, y_train_val,
        test_size=test_size/(train_size+validate_size), random_state=2021
    )
    (data_dir / 'train_text').write_text('\n'.join(X_train))
    (data_dir / 'validate_text').write_text('\n'.join(X_validate))
    (data_dir / 'test_text').write_text('\n'.join(X_test))
    (
        (data_dir / 'train_prices')
        .write_text('\n'.join(str(val) for val in y_train))
    )
    (
        (data_dir / 'validate_prices')
        .write_text('\n'.join(str(val) for val in y_validate))
    )
    (
        (data_dir / 'test_prices')
        .write_text('\n'.join(str(val) for val in y_test))
    )


def load_data(dataset, n_examples=None):
    """Return a tuple of (reviews, prices) for the given dataset in {'train',
    'validate', 'test'}. Only the first n_examples items of each will be
    returned (or all if n_examples=None). If the data doesn't exist it will
    be generated first

    """
    text_file_name = f'{dataset}_text'
    try:
        check_data_file(text_file_name)
    except RuntimeError:
        split_data()

    return (
        (data_dir / text_file_name).read_text().splitlines()[:n_examples],
        [
            int(x) for x in (
                (data_dir / f'{dataset}_prices').read_text()
                .splitlines()[:n_examples]
            )
        ]
    )
