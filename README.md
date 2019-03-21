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

``` 
brew install pipenv 
pipenv install 
```

This will install all of the dependencies/versions needed to execute the preprocessing and model training. 


#### Data prep 


First, convert the `.arff` to `.csv`. Not entirely necessary, but `.csv` is easier to work with. Do this with a simple script adapted from [here](https://github.com/Hutdris/arff2csv/blob/master/arff2csv.py). 

``` 
$ arff2csv.py 1year.arff 1year.csv 
``` 


Then, we can select any of the columns we need to make good predictions. Sometimes it is worth having all, but for simplicity, we'll just use the first 5. And include #65 since that one has our labels.  

``` 
$ cat 1year.csv | cut -d, -f 1,2,3,4,5,65 > 1year_cols.csv
``` 


#### Preprocessing 

We can pull apart the different steps involved in cleaning and separating the data into test and training sets. 

First to read the data: 

```
full = pd.read_csv(FULL_DATA, names=CSV_COLUMN_NAMES, header=0) 
```

Then remove oddball characters. This is always a problem, and one of the more manual aspects of data science. The offending entries in this dataset are question marks (`?`), for some strange reason. And the response column is a clean integer, so exclude it from the process. We can just remove them, because upon inspection, these tend to be empty rows altogether: 

```
for col in list(full)[:-1]: 
    print(col) 
    full = full[full[col] != '?']

```

Next split the dataset into test and train. Generally this should be done randomly, so the vertical structure doesn't introduce biases. For example, in this dataset, the bankrupt companies are all at the bottom of the file. So we need to randomize to ensure slightly better balance: 

```
is_train = np.random.uniform(0, 1, len(full)) <= .8
train, test = full[is_train == True], full[is_train == False]
```

Finally, we can pull the predictor vs response columns so they feed into the model the way TensorFlow wants them: 

```
train_x, train_y = train, train.pop(y_name)
test_x, test_y = test, test.pop(y_name)
```





























