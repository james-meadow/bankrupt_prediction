import pandas as pd
import numpy as np 
import tensorflow as tf


"""
Dataset from here: 
https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data# 

CSV prep: 
cat 1year.csv | cut -d, -f 1,2,3,4,5,65 > 1year_cols.csv
"""

FULL_DATA = "~/Desktop/data/1year_cols.csv"
TRAIN_DATA = "~/Desktop/data/1year_train.csv"
TEST_DATA = "~/Desktop/data/1year_test.csv"


CSV_COLUMN_NAMES = ['Attr1','Attr2','Attr3','Attr4','Attr5','class']
RESPONSE = [0, 1]


def treat_data(FULL_DATA, TRAIN_DATA, TEST_DATA): 
    """brings in full csv and removes bad values 
    returns new csvs in place using the paths above """
    full = pd.read_csv(FULL_DATA, names=CSV_COLUMN_NAMES, header=0) 
    
    for col in list(full)[:-1]: 
        print(col) 
        full = full[full[col] != '?']

    is_train = np.random.uniform(0, 1, len(full)) <= .8
    train, test = full[is_train == True], full[is_train == False]
    train.to_csv(TRAIN_DATA) 
    test.to_csv(TEST_DATA)


def load_data(y_name='class'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    # train_path, test_path = maybe_download()
    treat_data(FULL_DATA, TRAIN_DATA, TEST_DATA)

    train_path, test_path = TRAIN_DATA, TEST_DATA

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0) 
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    print(train_x.dtypes)
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

