
import tensorflow as tf

from canalflow.model import cnn_model_fn
from canalflow.preprocessing import input_from_dataset, load_test_set


def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    print("Installed version: " + str(tf.VERSION))


def main(_):
    installation_test()

    # test_input, test_labels = load_test_set()

    canalflow_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="../temp/cnn_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    canalflow_classifier.train(
        input_fn=input_from_dataset,
        steps=20000,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": test_input},
    #     y=test_labels,
    #     num_epochs=1,
    #     shuffle=False)
    # eval_results = canalflow_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)

    return


if __name__ == "__main__":
    tf.app.run()
