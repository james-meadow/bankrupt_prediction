#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import tensorflow as tf
from tensorflow.compat.v1.saved_model import simple_save
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
        hidden_units=[20, 20, 20],
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:bankrupt_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))


    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    # expected = ['Setosa', 'Versicolor', 'Virginica']
    # predict_x = {
    #     'Attr1': [0.12408, 0.014946, 0.090042, 0.12953, -0.018516, -0.000153, -0.33033, 0.13331, 0.006404, 0.019432, -0.051896, 0.018371],
    #     'Attr2': [0.83837, 0.94648, 0.20136, 0.35199, 0.32886, 0.50582, 0.93266, 0.16451, 0.67477, 0.61633, 0.5363, 0.4741],
    #     'Attr3': [0.14204, 0.03211, 0.41321, 0.57602, 0.069952, -0.42127, -0.47693, 0.40084, -0.008438, 0.057718, 0.054014, -0.13619],
    #     'Attr4': [1.1694, 1.03633, 1.2003, 2.6616, 1.257, 0.15367, 0.48863, 3.4366, 0.98668, 1.1062, 1.1061, 0.60839],
    #     'Attr5': [-91.88, -20.581, 50.125, -10.926, -29.298, -883.03, -350.83, 2.8373, -45.599, -162.88, -9.7422, -18.449]
    # }

    # predictions = classifier.predict(
    #     input_fn=lambda:bankrupt_data.eval_input_fn(predict_x, 
    #         labels=None, batch_size=args.batch_size))    

    raw_predictions = classifier.predict(
        input_fn=lambda:bankrupt_data.eval_input_fn(test_x, 
            labels=None, batch_size=args.batch_size))    
    predictions = [p['class_ids'][0] for p in raw_predictions]
    confusion_matrix = tf.confusion_matrix(list(test_y.tolist()), predictions)
    print(confusion_matrix)
    with tf.Session():
        print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))

    # for p in range(0,len(predict_x['Attr1'])): 
    #     print(next(predictions))


    # with tf.Session() as sess:
    #     simple_save(sess,
    #         '.',
    #         inputs={"x": x, "y": y},       #####################33add inputs 
    #         outputs={"z": z})

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
