import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils      

from google.cloud import storage





"""
Dataset from here: 
https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data# 

CSV prep: 
cat 1year.csv | cut -d, -f 2-10,65 > 1year_cols.csv
cat 2year.csv | cut -d, -f 1-10,65 > 2year_cols.csv
cat 1year.csv | cut -d, -f 1-10,65 > 3year_cols.csv

cat 1year.csv > all_years.csv 
tail -n +2 2year.csv >> all_years.csv 
tail -n +2 3year.csv >> all_years.csv
tail -n +2 4year.csv >> all_years.csv
tail -n +2 5year.csv >> all_years.csv

"""

# FULL_DATA = "~/Desktop/data/1year_cols.csv"
# FULL_DATA = "~/Desktop/data/1year.csv"

# FULL_DATA = "~/Desktop/data/all_years.csv"
# TRAIN_DATA = "~/Desktop/data/1year_train.csv"
# TEST_DATA = "~/Desktop/data/1year_test.csv"

PROJECT = 'cloud-academy'
BUCKET = "bankruptcy-prediction"
LOCAL = 'data/'
FULL_DATA = "all_years.csv"
TRAIN_DATA = LOCAL + "1year_train.csv"
TEST_DATA = LOCAL + "1year_test.csv"


CSV_COLUMN_NAMES = ['Attr{}'.format(i) for i in range(1,65)]
CSV_COLUMN_NAMES.append('class')
RESPONSE = [0, 1]

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    

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
    returns new csvs in place using the paths above """
    
    download_blob(bucket_name=BUCKET, 
                  source_blob_name='data/' + FULL_DATA, 
                  destination_file_name= LOCAL + FULL_DATA)

    full = pd.read_csv(LOCAL + FULL_DATA, names=CSV_COLUMN_NAMES, header=0) 
    
    for col in list(full)[:-1]: 
        full = full[full[col] != '?']

    is_train = np.random.uniform(0, 1, len(full)) <= .8
    train, test = full[is_train == True], full[is_train == False]
    train.to_csv(TRAIN_DATA) 
    test.to_csv(TEST_DATA)


def augment_data(x, y): 
    """Major balance problem, so jitter the numbers 
    to improve predictions. """
    
    print(1 - y.value_counts()[1] / y.value_counts()[0])
    these = y == 1
    
    aug_y = y[these]
    aug_x = x[these]

    aug_mult = 11
    l = len(x)
    nc = len(list(x))

    for i in range(aug_mult): 
        x = pd.concat([x, aug_x * np.random.uniform(0.8, 1.2, nc)])
        y = y.append(aug_y)

    # print(train_x)    
    # print(y)
    print(1 - y.value_counts()[1] / y.value_counts()[0])

    return x, y


def load_data(y_name='class'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
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
    return tf.estimator.export.ServingInputReceiver(features, 
                                                    feature_placeholders)


