

## Local train. 
# gcloud ml-engine local train \
#   --module-name=trainer.rnn_optimized \
#   --package-path=./trainer



## Cloud train 

## Before this runs, 
## must create bucket, 
## and run preprocessing script 
## and dump the pickles into the bucket. 
export BUCKET_NAME=bioreactor_prediction_data 
export JOB_NAME="rnn_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-west1                                      ###############. is this possible? 

gcloud ml-engine jobs submit training $JOB_NAME \
  --stream-logs \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --runtime-version 1.8 \
  --python-version 3.5 \
  --module-name trainer.rnn_optimized \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --training_set training_data \
  --job_name $JOB_NAME 

# gcloud ml-engine jobs stream-logs JOB $JOB_NAME 

