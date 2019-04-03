


##################################################
## Run this each time a new model is put into production
##   Change deploymentUri and version number
##   The last command will send a
##   single prediction to test the deployment
##################################################


MODEL_NAME=bankrupt_prediction
REGION=us-central1
BUCKET_NAME=bankrupt-prediction
#OUTPUT_PATH=gs://$BUCKET_NAME/model
deploymentUri=gs://$BUCKET_NAME/model/1554098440/
VERSION_NAME=v3
INPUT_FILE=test_prediction.json

## create model placeholder
## only use if creating a completely new named model
# gcloud ml-engine models create $MODEL_NAME --regions=$REGION

## list the bucket
#gsutil ls -r $OUTPUT_PATH
#gsutil ls -r $deploymentUri

## create a new version of an existing model
## Only use if deploying a new version
# gcloud ml-engine versions create $VERSION_NAME \
#     --model $MODEL_NAME \
#     --origin $deploymentUri \
#     --runtime-version 1.13
#     --python-version=2.7

## Send a prediction using a json file.
gcloud ml-engine predict \
    --model $MODEL_NAME \
    --version $VERSION_NAME \
--json-instances $INPUT_FILE
