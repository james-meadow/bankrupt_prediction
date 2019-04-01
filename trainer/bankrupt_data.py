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

# def serving_input_fn():
#     feature_placeholders = {
#         'id': tf.placeholder(tf.string, [None], name="id_placeholder"),
#         'feat': tf.placeholder(tf.float32, [None, 64], name="feat_placeholder"),
#     }
#     # return feature_placeholders 
#     return input_fn_utils.InputFnOps(
#         feature_placeholders,
#         None,
#         feature_placeholders)


# feature_placeholders = {}
# keys = CSV_COLUMN_NAMES[:-1]
# values = tf.placeholder(tf.float32, [None])
# for i in keys:
#     feature_placeholders[i] = values  

# keys = ['Attr1','Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8','Attr9','Attr10','Attr11','Attr12','Attr13','Attr14','Attr15','Attr16','Attr17','Attr18','Attr19','Attr20','Attr21','Attr22','Attr23','Attr24','Attr25','Attr26','Attr27','Attr28','Attr29','Attr30','Attr31','Attr32','Attr33','Attr34','Attr35','Attr36','Attr37','Attr38','Attr39','Attr40','Attr41','Attr42','Attr43','Attr44','Attr45','Attr46','Attr47','Attr48','Attr49','Attr50','Attr51','Attr52','Attr53','Attr54','Attr55','Attr56','Attr57','Attr58','Attr59','Attr60','Attr61','Attr62','Attr63','Attr64']
# values = [0.20055,0.37951,0.39641,2.0472,32.351,0.38825,0.24976,1.3305,1.1389,0.50494,0.24976,0.6598,0.1666,0.24976,497.42,0.73378,2.6349,0.24976,0.14942,43.37,1.2479,0.21402,0.11998,0.47706,0.50494,0.60411,1.4582,1.7615,5.9443,0.11788,0.14942,94.14,3.8772,0.56393,0.21402,1.741,593.27,0.50591,0.12804,0.66295,0.051402,0.12804,114.42,71.05,1.0097,1.5225,49.394,0.1853,0.11085,2.042,0.37854,0.25792,2.2437,2.248,348690,0.12196,0.39718,0.87804,0.001924,8.416,5.1372,82.658,4.4158,7.4277]
# FEATURES = dict(zip(keys, values))

# def serving_input_fn():
#     inputs = FEATURES
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)
#     return dataset 



## still need to deploy and test this with json. 
## now each placeholder has a distinct name
## hopefully that solves 1 error. 

def serving_input_fn():
    # feature_placeholders = {'attr': tf.placeholder(tf.float32, [None])}
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


