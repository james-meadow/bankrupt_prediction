

"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.utils import input_fn_utils      
import bankrupt_data



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

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
        model_dir='model/')

    # Train the Model.
    classifier.train(
        input_fn=lambda:bankrupt_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    
    ###############################################
    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    raw_predictions = classifier.predict(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x, 
            labels=None, batch_size=args.batch_size))    
    predictions = [p['class_ids'][0] for p in raw_predictions]
    confusion_matrix = tf.confusion_matrix(list(test_y.tolist()), predictions)
    with tf.Session():
        print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))


    ###############################################
    ## Export the model 
    classifier.export_savedmodel(
        "model-export", 
        bankrupt_data.serving_input_fn) 
        # bankrupt_data.eval_input_fn) 



if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
