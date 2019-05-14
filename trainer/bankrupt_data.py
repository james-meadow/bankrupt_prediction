import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO
from google.cloud import storage



"""
Dataset from here:
https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#


"""

BUCKET = "bankrupt-prediction"
# LOCAL = os.path.join(tempfile.gettempdir(), 'data')
FULL_DATA = "all_years.csv"
TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"


CSV_COLUMN_NAMES = ['Attr{}'.format(i) for i in range(1,65)]
CSV_COLUMN_NAMES.append('class')
RESPONSE = [0, 1]
#
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

#

def read_data(gcs_path):
   print('downloading csv file from', gcs_path)
   file_stream = file_io.FileIO(gcs_path, mode='r')
   data = pd.read_csv(StringIO(file_stream.read()))
   return data

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def treat_data(BUCKET, FULL_DATA, TRAIN_DATA, TEST_DATA):
    """brings in full csv and removes bad values
    returns new csv in place using the paths above """
    # download_blob(bucket_name=BUCKET,
    #               source_blob_name='data/' + FULL_DATA,
    #               destination_file_name= LOCAL + FULL_DATA)
    full = read_data('gs://bankrupt-prediction/data/all_years.csv')
    # full = pd.read_csv(LOCAL + FULL_DATA, names=CSV_COLUMN_NAMES, header=0)
    for col in list(full)[:-1]:
        try:
            full = full[full[col] != '?']
        except:
            print('{} has all numeric values'.format(col))

    is_train = np.random.uniform(0, 1, len(full)) <= .8
    train, test = full[is_train == True], full[is_train == False]
    train.to_csv(TRAIN_DATA)
    test.to_csv(TEST_DATA)


def augment_data(x, y):
    """Major unbalanced dataset, so jitter the numbers
    to improve predictions. """
    ## what percent are bankruptcies
    print(1 - y.value_counts()[1] / len(y))

    ## select the low-n class
    these = y == 1
    aug_y = y[these]
    aug_x = x[these]

    ## create a multiplier.
    ## This * n of the lower class results in new, more balanced dataset.
    aug_mult = 35
    # aug_mult = 11
    nc = len(list(x))

    for i in range(aug_mult):
        x = pd.concat([x, aug_x * np.random.uniform(0.5, 1.5, nc)])
        y = y.append(aug_y)

    print(1 - y.value_counts()[1] / len(y))

    return x, y


def load_data(y_name='class'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""
    # train_path, test_path = maybe_download()
    treat_data(BUCKET, FULL_DATA, TRAIN_DATA, TEST_DATA)

    train_path, test_path = TRAIN_DATA, TEST_DATA

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    train_x, train_y = augment_data(train_x, train_y)
    test_x, test_y = augment_data(test_x, test_y)

    return (train_x, train_y), (test_x, test_y)



def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def serving_input_fn():
    feature_placeholders = {}
    keys = CSV_COLUMN_NAMES[:-1]

    for i in keys:
        feature_placeholders[i] = tf.placeholder(tf.float32, [None])
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
