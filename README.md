## Tensorflow Estimator Example

#### James Meadow, Jason Shi, Maryam Veisi 
#### 6/4/2019

*Demonstrates an example classification pipeline to use a pre-built TensorFlow Estimators (DNN).*

----------

### Data Gathering and Runbook

*This section holds the commands to quickly treat data, and to compile, train, evaluate, and send a prediction. Detailed breakdown is below in the `Details...` section*

#### Dataset: Company bankruptcy predictions.

* Full dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data#)
* Download all dataset files to work through this exercise
* Has 60+ numeric features associated with finances of thousands of Polish companies.
* This model implementation uses all years of the dataset to predict which of the companies will go bankrupt during the study.
* Original data was in the `.arff` format, which was transformed to `.csv` using a script below.


#### Environment Setup
Install virtual environment if you haven't.

```bash
pip install virtualenv
```

Assuming you have installed virtualenv, you can create and activate a new python 3 virtual environment:

```bash
virtualenv —python python3 env
source env/bin/activate
pip install -r requirements.txt
```
where 'env' is the name of the environment.

The second line will active the virtual environment and this third line will install all of the dependencies/versions needed to execute this example preprocessing and model training.

Also we need to install the latest version of Cloud SDK using the following command:
```bash
curl https://sdk.cloud.google.com | bash
```

#### Data prep

First, convert the `.arff` to `.csv`. Not entirely necessary, but `.csv` is easier to work with. Do this with a simple script adapted from [here](https://github.com/Hutdris/arff2csv/blob/master/arff2csv.py).


```bash
$ arff2csv.py 1year.arff 1year.csv
...
```

Then we can pile them all together. This will create a dataset of companies that either did or did not go bankrupt during a 5 year study.

```bash
$ cat 1year.csv > all_years.csv
$ tail -n +2 2year.csv >> all_years.csv
$ tail -n +2 3year.csv >> all_years.csv
$ tail -n +2 4year.csv >> all_years.csv
$ tail -n +2 5year.csv >> all_years.csv
```

This can now be saved in a GCS bucket and be called when our model trains.

Of course you'll need to use a GCS bucket that you can access and a GCP project with your permissions. You can change those parameters in the top of the `bankrupt_data.py` file.


#### Runbook


Assuming the data are already treated (as above in the Data Prep section), the primary python script is `premade_estimator_bankrupt.py`. This will call a handfull of functions from the `bankrupt_data.py` script.

Here is the one-liner that will kick off the whole process (preprocessing --> training --> validating --> testing --> saving)

```
python3 ./premade_estimator_bankrupt.py
```


If all is set up correctly, this should show you a verbose output while it compiles, trains, and evaluates the model.

When the model is optimized to your liking, we can deploy to CMLE by selecting the correct inside `deploy_cmle.sh`:

```bash
MODEL_NAME=bankrupt_prediction
REGION=us-central1
BUCKET_NAME=bankrupt-prediction
OUTPUT_PATH=gs://$BUCKET_NAME/model
MODEL_BINARIES=gs://$BUCKET_NAME/model/1554098440/

## create model in CMLE
gcloud ml-engine models create $MODEL_NAME --regions=$REGION

## create a new version of an existing model
gcloud ml-engine versions create v3 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.13
```

Then a single command sends data to the model to the CMLE endpoint for a new prediction. The `test_prediction.json` file was created for demo purposes by the `create_single_prediction_json.py` file.

```bash
gcloud ml-engine predict \
    --model $MODEL_NAME \
    --version v3 \
    --json-instances test_prediction.json
```

You should get a response that looks something like this, which shows that this company will not go bankrupt during the study, which is correct!  

```
CLASS_IDS  CLASSES  LOGISTIC  LOGITS                 PROBABILITIES
[0]        [u'0']   [0.0]     [-265.59613037109375]  [1.0, 0.0]
```



-----------




### Details of Data Preprocessing, Modeling, and Deployment

*Everything below this line is just an explanation of the code that already executed during the Runbook section above.*

#### Preprocessing

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

*Note there is another interesting approach to identifying and removing these funky values in the `data_exploration.ipynb` script. The version above was used in the official preprocessing workflow.* 


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

Since this is a highly unbalanced dataset (and the small class is insufficiently small), we'll have to do a bit of data augmentation to expand the smallest class. Not ideal, and there are lots of other ways to augment data, but we can use this quick and dirty method to our advantage as we optimize parameters. In this case, I'll just add a small amount of random noise to expand the bankrupt companies. 

*Note there is an exploration of the `SMOTE` function for balancing in `data_exploration.ipynb`, but it did not improve performance of the model, so the version below is used in this first take on the model.* 

```bash
these = y == 1
aug_y = y[these]
aug_x = x[these]
x = pd.concat([x, aug_x * np.random.uniform(0.8, 1.2, nc)])
y = y.append(aug_y)
```

This can be iterated with variable balance rates and random noise until a useful balance can be achieved. We found that these variables captured most of the performance variability we see in the model, so this is a great lever to get the model to put out the right results. 


We also explored removing some of the more highly-correlated variables (of which there are lots!), but found that this did not effectively improve model performance. This makes sense since DNN models are not terribly sensitive to correlation within features. The model tends to learn to ignore anything that doesn't help. 

```python 
def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.1, annot=False, cbar_kws={"shrink": .50})
    plt.show();
    
correlation_heatmap(data)
```

![full-correlation](figs/full_corr.png)

Since there are lots of correlations in this dataset, let's see whether we can influence model fit by removing them. 

```python
# Create correlation matrix
corr_matrix = refined_data.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.55)]
refined_data.drop(to_drop, axis=1, inplace=True)
```

We've now removed many columns and the result is a much less highly correlated feature set: 

```python
correlation_heatmap(refined_data)
```

![reduced_correlation](figs/reduced_corr.png)

And we can check out which columns were cut. Note the dataset is annoyingly labeled with 'Attr1' etc for the feature names. The original names are available with the data source (linked at the top of this document, and captured in `feature_names.txt`), so here is a quick lookup function: 

```python 
def feature_name(st): 
    i = int(st.strip('Attr'))
    return feature_names[i-1]

cut_cols = ['Attr3', 'Attr6', 'Attr8', 'Attr10', 'Attr11', 'Attr14'] 
[print(feature_name(f)) for f in cut_cols]
```

```
working capital / total assets
retained earnings / total assets
book value of equity / total liabilities
equity / total assets
(gross profit + extraordinary items + financial expenses) / total assets
(gross profit + interest) / total assets
```

Well these actually make sense. If we look back at the column names that were kept, these are generally derivative of the kept columns. No harm in cutting them. 

We can pass both through the model and find out if this improves our results (update: it doesn't change much). 


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

We also need a function to tell the model what to expect after it has been deployed to CMLE. This can be done by creating a structure of `tf.placeholder`s for each feature the model should expect:

```python  
feature_placeholders = {}
keys = CSV_COLUMN_NAMES[:-1]

for i in keys:
    feature_placeholders[i] = tf.placeholder(tf.float32, [None])
features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
}
```


#### Estimator function

This will all be put inside of a function, so see `premade_estimator_bankrupt.py` for the flow of data. Here are the relevant parts.

We'll use the pre-build TensorFlow `DNNClassifier` model, with a simple set of parameters since this is a relatively simple dataset:


```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[20, 20, 20],
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



#### Hyperparameter Tuning

HP tuning is an important part of any Deep Learning project. We can test the model performance against a long list of potential parameters. For example, we can instruct the model to test a variety of batch sizes to determine if that incluences the fit. For this example, we'll test `batch_size` and `learning_rate` as a simple demonstration of CMLE HP tuning. 

Using CMLE (ai-platform) to test hyperparameters is fairly straightforward, but you'll need to be careful to set up the config file correctly. 

Hyperparameter configurations in this example are stored in `hptuning_config.yaml`. This job is just named 'job1', and that name can be used to check logs during and after tuning. The yaml format lays out the parameters we want to test. Clearly this is a small subset of potential hyperparameters and ranges, but we'll just focus on a simple example for now: 

```bash 
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: epoch_acc
    maxTrials: 4
    maxParallelTrials: 2
    params:
    - parameterName: batch_size
      type: INTEGER
      minValue: 100
      maxValue: 102
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.01
      maxValue: 0.1
      scaleType: UNIT_LOG_SCALE
```




Once the yaml config file is sorted out, you can start the job in gcloud using the following command (which is in `hp_tune.sh`):

```bash
gcloud ai-platform jobs submit training job1 \
  --config hptuning_config.yaml \
  --package-path trainer/ \
  --module-name trainer.premade_estimator_bankrupt \
  --region us-central1 \
  --python-version 3.5 \
  --runtime-version 1.13 \
  --job-dir=gs://bankrupt-prediction/model/train \
  --stream-logs
```

In this example, the range of hyperparameters didn't make a huge difference, but for this model, `batch_size=100` and `learning_rate=0.04` give us a slightly more balanced confusion matrix. In the Evaluation section, we'll discuss the goals in a bit more detail, but we're not interested in perfect accuracy -- bankruptcy is an incredibly difficult thing to predict. Rather we're interested in raising a red flag on companies that look like they are at risk. We can use this to evaluate things like investments and forecast revenue. Thus we see that 1/2 of companies in the evaluation dataset look to be at risk. Great first step, so now on to train the model using these updated parameters. 


#### Evaluation

At the end of training, we can use the randomly excluded dataset (`test_x` and `test_y`) to gauge the accuracy of the model. But accuracy is not the only thing to optimize. In this example, it is *really really* difficult, if not impossible to accurately predict which companies will go bankrupt just looking at their financials. So 100% accuracy is neither achievable, nor is it the right benchmark for us. Instead, we can target high-risk companies: Those that are predicted to go bankrupt even though they don't during this particular study. Therefore we can optimize for true positives and also false positives. Those false positives are the ones that look like they might go bankrupt but just make it through the study.

In other words, one potential way to optimize is to try to achieve: `true > false positive > false negative`. This might result in low accuracy, but accuracy is sometimes the wrong benchmark, depending on the goal of a model.

First, we want to see just an aggregated accuracy score:

```python
eval_result = classifier.evaluate(
    input_fn=lambda:bankrupt_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

But I'm more interested in how the model deals with those that are at risk of bankruptcy, so we want to explore the confusion matrix:

```python
raw_predictions = classifier.predict(
    input_fn=lambda:bankrupt_data.eval_input_fn(test_x,
        labels=None, batch_size=args.batch_size))    
predictions = [p['class_ids'][0] for p in raw_predictions]
confusion_matrix = tf.confusion_matrix(list(test_y.tolist()), predictions)
with tf.Session():
    print('\nConfusion Matrix:\n',
          tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))
```

Depending on how we want to use these predictions, we can either optimize for high accuracy, or allow lower accuracy and generate a risk score for companies that look like they might go bankrupt.

#### Performance 

As explained above, sometimes the goal is not strictly accuracy, but rather identifying high-risk targets. In this case, we want to identify companies who's financials resemble those of bankrupted companies. Thus let's see how the model performs. This output is from the confusion matrix command above.  


```
Test set accuracy: 0.565

Confusion Matrix:
 [[2568 1319]
 [1861 1559]]
```

So yes, very low accuracy, but we're actually leaning in the right direction for identifying high risk profiles. We've almost achieved our goal to identify correct=57%,  at-risk=18%, & incorrect=25%. 

Now we can focus on those companies that are not bankrupt yet, but are predicted to be at risk. Thus one potential follow-up step in this project could be to analyze their financial profiles to evaluate which features are the strongest indicators for bankruptcy, and then dig in with subject matter experts to design a risk-analysis framework. There is lots of additional optimization and exploration that can enhance these results, but that's for another day! 




#### Deploy to CMLE

After you have trained your model, you must make important adjustments before deploying it to Cloud ML Engine for predictions.

1. Export your model to a SavedModel that can be deployed to Cloud ML Engine.
2. Ensure that the file size of your SavedModel is under the Cloud ML Engine default limit of 250 MB by exporting a graph specifically for prediction.

To put this model into production, we first save to a format that CMLE can easily take in, and use our `serving_input_fn()` as the input placeholder:

```python
classifier.export_savedmodel(
    "model-export",
    bankrupt_data.serving_input_fn)
```


Then we need to upload the exported model to the Cloud Storage bucket. Run the following command to upload your saved model to your bucket in Cloud Storage:

```bash
SAVED_MODEL_DIR=$(ls ./[your-export-dir-base] | tail -1)
gsutil cp -r $SAVED_MODEL_DIR gs://[your-bucket]
```

Next the model gets deployed using the `gcloud` command line tool:

```bash
MODEL_NAME=bankrupt_prediction
REGION=us-central1
BUCKET_NAME=bankrupt-prediction
OUTPUT_PATH=gs://$BUCKET_NAME/model
DEPLOYMENT=gs://$BUCKET_NAME/model/1554098440/
VERSION_NAME=v3
INPUT_FILE=test_prediction.json

## create model placeholder
gcloud ml-engine models create $MODEL_NAME --regions=$REGION

## create a new version of an existing model
gcloud ml-engine versions create $VERSION_NAME \
    --model $MODEL_NAME \
    --origin $DEPLOYMENT \
    --runtime-version 1.13
```


Keep in mind that MODEL_NAME must be unique within the Cloud ML Engine model.

If all goes well, there is now a CMLE API awaiting new data in json format. You can call this from pretty much anywhere as long as the incoming data are in the correct format.

#### Serve predictions

There are lots of ways to serve models in CMLE, but we'll just use the single prediction method, since that's what the input function expects.

```bash
gcloud ml-engine predict \
    --model $MODEL_NAME \
    --version $VERSION_NAME \
--json-instances $INPUT_FILE
```
