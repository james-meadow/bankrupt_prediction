
## Run HP tuning on GCP CMLE 
## This uses the `hptuning_config.yaml` file as setup 
gcloud ai-platform jobs submit training job1 \
  --config hptuning_config.yaml \
  --package-path trainer/ \
  --module-name trainer.premade_estimator_bankrupt \
  --region us-central1 \
  --python-version 3.5 \
  --runtime-version 1.13 \
  --job-dir=gs://bankrupt-prediction/model/train \
  --stream-logs