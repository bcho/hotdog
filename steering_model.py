"""steering angle model

ref: https://github.com/commaai/research/blob/master/train_steering_model.py
"""

import argparse
import typing as T

import keras
from keras.layers.convolutional import Convolution2D
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

from hotdog import image_data
from hotdog.config import Config
from hotdog.image import normalize_image
from hotdog.logging import make_logger


MODEL_NAME = 'steering'
HOTDOG_LABEL = 'category=hotdog'


logger = make_logger(MODEL_NAME, 'info')


def load_train_data(
    config: Config,
    train_size: int,
    image_size: T.Tuple[int, int],
    train_set_split_state: T.Optional[int] = None,
):
    images = image_data.load(config, image_data.prefix_label)
    positive = image_data.augment(
        [i for i in images if HOTDOG_LABEL in i.labels],
        count=train_size,
        image_size=image_size,
    )
    negative = image_data.augment(
        [i for i in images if HOTDOG_LABEL not in i.labels],
        count=train_size,
        image_size=image_size,
    )
    if train_set_split_state is None:
        train_set_split_state = np.random.randint(0, 100)
    logger.info(f'train data split state: {train_set_split_state}')

    X = np.array([i.load_image_data() for i in positive + negative])
    y = [1] * len(positive) + [0] * len(negative)
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=train_set_split_state
    )

    logger.debug(f'X_train shape: {X_train.shape}')
    logger.debug(f'X_test shape: {X_test.shape}')
    logger.debug(f'Y_train shape: {y_train.shape}')
    logger.debug(f'Y_test shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test


def get_model(input_shape):
    model = keras.Sequential()

    model.add(Convolution2D(
        16, (8, 8),
        strides=(4, 4),
        padding='same',
        input_shape=input_shape
    ))
    model.add(keras.layers.ELU())

    model.add(Convolution2D(
        32, (5, 5),
        strides=(2, 2),
        padding='same',
    ))
    model.add(keras.layers.ELU())

    model.add(Convolution2D(
        64, (5, 5),
        strides=(2, 2),
        padding='same',
    ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.ELU())

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.ELU())

    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Activation('softmax'))

    # ref: https://keras.io/models/model/#compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train(args: argparse.Namespace):
    config = args.config
    image_size = (args.image_size_width, args.image_size_height)

    logger.info(f'train with {args.train_size} {image_size}')

    X_train, X_test, y_train, y_test = load_train_data(
        config=config,
        train_size=args.train_size,
        image_size=image_size,
        train_set_split_state=(args.train_set_state),
    )
    model = get_model(image_size + (1, ))
    model.fit(
        X_train,
        y_train,
        nb_epoch=args.epoch,
        validation_split=.1
    )

    metrics = model.evaluate(X_test, y_test)
    for metric_name, metric_value in zip(model.metrics_names, metrics):
        logger.info(f'{metric_name}: {metric_value}')

    config.ensure_data_paths()
    model_path = config.get_model_path(f'{MODEL_NAME}.h5')
    model.save(model_path)
    logger.info(f'model saved to {model_path}')


def predict(args: argparse.Namespace):
    import cv2

    config = args.config

    config.ensure_data_paths()
    model_path = config.get_model_path(f'{MODEL_NAME}.h5')
    model = load_model(model_path)
    logger.debug('model loaded from {model_path}')

    # NOTE: batch_size, width, height, _
    image_size = (model.input_shape[1], model.input_shape[2])

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image = normalize_image(image, image_size)
    p = model.predict(np.array([image]))[0]

    print(f'image: {args.image_path}')
    print(f'is a hotdog: {p[1]}')
    print(f'is not a hotdog: {p[0]}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description='steering angle model')

    cli.set_defaults(config=Config)
    cli.set_defaults(func=lambda x: cli.print_usage())

    subparsers = cli.add_subparsers(help='sub commands')

    cli_train = subparsers.add_parser('train', help='train model')
    cli_train.add_argument(
        '--train_size', type=int, default=15000,
        help='size of train set',
    )
    cli_train.add_argument(
        '--image_size_width', type=int, default=128,
        help='image width',
    )
    cli_train.add_argument(
        '--image_size_height', type=int, default=128,
        help='image height',
    )
    cli_train.add_argument(
        '--train_set_state', type=int, default=None,
        help='train set split random state, defaults to random',
    )
    cli_train.add_argument(
        '--epoch', type=int, default=10,
        help='number of epoch',
    )
    cli_train.set_defaults(func=train)

    cli_predict = subparsers.add_parser('predict', help='predict with model')
    cli_predict.add_argument('image_path', type=str, help='path to the image')
    cli_predict.set_defaults(func=predict)

    args = cli.parse_args()
    args.func(args)
