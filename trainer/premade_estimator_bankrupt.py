

"""Defines a DNNClassifier for the bankrupcy dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
import bankrupt_data


#
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# parser.add_argument('--train_steps', default=1000, type=int,
#                     help='number of training steps')

def get_args():
  """Argument parser.

  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir',
      # default=
      type=str,
      # required=True,
      help='local or GCS location for writing checkpoints and exporting models')
  # parser.add_argument(
  #     '--num-epochs',
  #     type=int,
  #     default=20,
  #     help='number of times to go through the data, default=20')
  parser.add_argument(
      '--batch_size',
      default=10,
      type=int,
      help='number of records to read during each training step, default=128')
  parser.add_argument(
      '--learning_rate',
      default=.01,
      type=float,
      help='learning rate for gradient descent, default=.01')
  args, _ = parser.parse_known_args()
  return args

def main(args):
    # args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = bankrupt_data.load_data()


    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # hidden_units=[64, 64, 64],
        hidden_units=[20, 20, 20],
        n_classes=2,
        # hidden_units=[10, 10],
        optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=args.learning_rate,
        l1_regularization_strength=0.001
        ))

    # Train the Model.
    classifier.train(
        input_fn=lambda:bankrupt_data.train_input_fn(train_x, 
          train_y,
          100),
        steps=100)


    ###############################################
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x, test_y,
                                                100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    ###############################################
    # Predict and print the confusion matrix.
    raw_predictions = classifier.predict(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x,
            labels=None, batch_size=args.batch_size))
    predictions = [p['class_ids'][0] for p in raw_predictions]
    confusion_matrix = tf.confusion_matrix(list(test_y.tolist()), predictions)
    with tf.Session():
        print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))


    ###############################################
    # Export and save the model.
    classifier.export_savedmodel(
        'gs://bankrupt-prediction/model/train/model-export',
        bankrupt_data.serving_input_fn)
        # bankrupt_data.eval_input_fn)




if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(tf.logging.ERROR)
    main(args)
