from collections import namedtuple

from sklearn.model_selection import train_test_split

from tensorflow.python.estimator.estimator_lib import TrainSpec
from tensorflow.python.estimator.estimator_lib import EvalSpec
from tensorflow.python.estimator.inputs.numpy_io import numpy_input_fn

from tensorflow.keras.datasets import mnist


class MNISTData(namedtuple('Data', ['batch_size', 'epochs'])):
    """
    A wrapper class for MNIST data
    """
    def __init__(self):
        """
        Downloads MNIST data, creates train/eval and test sets and
        defines feeder functions for TensorFlow Estimator
        """
        super(MNISTData, self).__init__()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, y_train, x_eval, y_eval = train_test_split(x_train, y_train,
                                                            test_size=0.1)
        self.train_input_fn = numpy_input_fn(x_train, y_train,
                                             batch_size=self.batch_size,
                                             num_epochs=self.epochs,
                                             shuffle=True)

        self.eval_input_fn = numpy_input_fn(x_eval, y_eval,
                                            batch_size=self.batch_size,
                                            num_epochs=1,
                                            shuffle=True)

        self.test_input_fn = numpy_input_fn(x_test, y_test,
                                            batch_size=self.batch_size,
                                            num_epochs=1,
                                            shuffle=True)

    @property
    def train_spec(self):
        """
        Train Spec
        :return: returns a Train Spec from train inputs
        """
        return TrainSpec(self.train_input_fn)

    @property
    def eval_spec(self):
        """
        Eval Spec
        :return: returns a Eval Spec from eval inputs
        """
        return EvalSpec(input_fn=self.eval_input_fn,
                        throttle_secs=0, start_delay_secs=0)
