MODEL_DIR="gs://bankrupt-prediction/model/1554098440/"
INPUT_FILE="test_prediction.json"
FRAMEWORK="TENSORFLOW"

gcloud ml-engine local predict --model-dir=$MODEL_DIR \
    --json-instances $INPUT_FILE \
    --framework $FRAMEWORK \
#    --verbosity debug
