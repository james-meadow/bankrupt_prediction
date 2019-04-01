


MODEL_NAME=bankrupt_prediction 
REGION=us-central1 
BUCKET_NAME=bankrupt-prediction
OUTPUT_PATH=gs://$BUCKET_NAME/model
# MODEL_BINARIES=gs://$BUCKET_NAME/model/model/
# MODEL_BINARIES=gs://$BUCKET_NAME/model/1554057935/
MODEL_BINARIES=gs://$BUCKET_NAME/model/1554098440/

## create model placeholder 
gcloud ml-engine models create $MODEL_NAME --regions=$REGION

## list the bucket 
gsutil ls -r $OUTPUT_PATH
gsutil ls -r $MODEL_BINARIES

## create a new version of an existing model 
gcloud ml-engine versions create v3 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.13

## Send a prediction using a json file. 
gcloud ml-engine predict \
    --model $MODEL_NAME \
    --version v3 \
    --json-instances test_prediction.json


## currently stopped here. not sure what the model wants for input 
# (p3env) S02233-MBPR:bankrupt_prediction james.meadow$ gcloud ml-engine predict     --model $MODEL_NAME     --version v1     --json-instances test_prediction.json
# {
#   "error": "Prediction failed: Error during model execution: AbortionError(code=StatusCode.INVALID_ARGUMENT, details=\"Endpoint \"Placeholder:0\" fed more than once.\")"
# }


