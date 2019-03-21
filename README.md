## Tensorflow Estimator Example 

#### James Meadow 
#### 3/20/2019 

*Demonstrates an example pipeline to use one of the pre-built TensorFlow Estimators (DNN).*

----------






#### Dataset: Company bankruptcy predictions. 

* Full dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#) 
* Has 60+ numeric features associated with finances of thousands of Polish companies. 
* This model implementation uses the `1year` dataset to predict which of the companis will go bankrupt in the first year of the study. 
* Original data was in the `.arff` format, which was transformed to `.csv` 



#### Environment 

Assuming you have installed Homebrew or Linuxbrew simply run:

```bash 
brew install pipenv 
pipenv install 
```

This will install all of the dependencies/versions needed to execute this example preprocessing and model training. 

If you are starting a new `pipenv` project, you can install packages like this: 

```bash 
pipenv install tensorflow
```

Etc. In this exercise, you'll need `tensorflow`, `pandas`, and `numpy`. But you don't have to worry about that if you just utilize the `pipenv install` command above. 


#### Data prep 


First, convert the `.arff` to `.csv`. Not entirely necessary, but `.csv` is easier to work with. Do this with a simple script adapted from [here](https://github.com/Hutdris/arff2csv/blob/master/arff2csv.py). 

```bash 
$ arff2csv.py 1year.arff 1year.csv 
``` 


Then, we can select any of the columns we need to make good predictions. Sometimes it is worth having all, but for simplicity, we'll just use the first 5. And include #65 since that one has our labels.  

```bash
$ cat 1year.csv | cut -d, -f 1,2,3,4,5,65 > 1year_cols.csv
``` 

These can now be saved in a GCS bucket and be called when our model trains. 

#### Preprocessing 

We can pull apart the different steps involved in cleaning and separating the data into test and training sets. 

First to read the data: 

```python
full = pd.read_csv(FULL_DATA, names=CSV_COLUMN_NAMES, header=0) 
```

Then remove oddball characters. This is always a problem, and one of the more manual aspects of data science. The offending entries in this dataset are question marks (`?`), for some strange reason. And the response column is a clean integer, so exclude it from the process. We can just remove them, because upon inspection, these tend to be empty rows altogether: 

```python
for col in list(full)[:-1]: 
    print(col) 
    full = full[full[col] != '?']

```

Next split the dataset into test and train. Generally this should be done randomly, so the vertical structure doesn't introduce biases. For example, in this dataset, the bankrupt companies are all at the bottom of the file. So we need to randomize to ensure slightly better balance: 

```python
is_train = np.random.uniform(0, 1, len(full)) <= .8
train, test = full[is_train == True], full[is_train == False]
```

Finally, we can pull the predictor vs response columns so they feed into the model the way TensorFlow wants them: 

```python
train_x, train_y = train, train.pop(y_name)
test_x, test_y = test, test.pop(y_name)
```

#### TF Input functions 

We need to feed data into our Tensorflow model in multiple formats. First to train and another to evaluate the trained model performance. 

First to train, we convert into a tf.Dataset. The second line shuffles the batches of data each time. 

```python
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) 
dataset = dataset.shuffle(1000).repeat().batch(batch_size)
```

The second input function can send either evaluation data or unlabeled data to predict: 

```python
features=dict(features)
if labels is None:
    inputs = features
else:
    inputs = (features, labels)

dataset = tf.data.Dataset.from_tensor_slices(inputs)
```

#### Estimator function 

This will all be put inside of a function, so see `premade_estimator_bankrupt.py` for the flow of data. Here are the relevant parts. 

We'll use the pre-build TensorFlow `DNNClassifier` model, with a simple set of parameters since this is a relatively simple dataset: 


```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=2)
```

The model is trained by passing in the input functions from before.  

```python 
(train_x, train_y), (test_x, test_y) = bankrupt_data.load_data()
classifier.train(
    input_fn=lambda:bankrupt_data.train_input_fn(train_x, train_y,
                                             args.batch_size),
    steps=args.train_steps)
```

The evaluation happens in a similar way: 

```python 
classifier.evaluate(
    input_fn=lambda:bankrupt_data.eval_input_fn(test_x, test_y,
                                            args.batch_size))
```


And voila! A *relatively* simple example showing how to harness the power of the pre-built TensorFlow Estimator models. 


#### Runbook 

To actually run the model, of course, one shouldn't just paste in all these blocks. There are lots of options, but here are the instructions to run a packaged version: 

Assuming the data are already treated (as above in the Data Prep section), the primary python script is `premade_estimator_bankrupt.py` and it will run all of the pre-processing steps contained in the `bankrupt_data.py` script. 

```
pipenv run python premade_estimator_bankrupt.py 
```

If all is set up correctly, this should show you a verbose output while it compiles, trains, and evaluates the model. 

From here, you can choose to save the model in many different formats, or deploy it with Cloud ML Engine as an endpoint to be called by a separate service. For example, if the data are streaming into BigQuery, you can write a simple Cloud Function script that queries new data, format it appropriately, and then send it to the endpoint for a fresh set of predictions. 

Next in this series, we'll look at some options for deploying this model in production! 



























