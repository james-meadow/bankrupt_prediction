import tensorflow as tf
# from tensorflow.contrib import predictor

model_dir = 'model/'
# predict_fn = tf.predictor.from_saved_model(model_dir)

with tf.Session() as session:
    meta_graph_def = tf.saved_model.loader.load(
    session,
    # tags=[tag_constants.SERVING],
    export_dir=model_dir 
    )
print(meta_graph_def)
