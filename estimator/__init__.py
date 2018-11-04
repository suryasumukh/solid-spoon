from tensorflow.python.estimator.estimator_lib import Estimator as TFEstimator
from tensorflow.python.estimator.estimator_lib import RunConfig


from utils import function_from_file
from utils import load_configuration


class configurable(object):
    """
    A decorator class to load configuration from YAML spec
    and return an instance of TensorFlow Estimator

    Returns: Estimator (A subclass of the TensorFlow Estimator)
    """
    def __init__(self, klass):
        self.klass = klass

    def __call__(self):
        yaml_spec = load_configuration()
        return self.klass(yaml_spec)


@configurable
class EstimatorFromYAML(TFEstimator):
    """
    An Estimator to train and evaluate a user defined TensorFlow graph

    This class is a subclass of the TensorFlow Estimator
    Refer: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/estimator/estimator.py
    """
    def __init__(self, yaml_spec):
        """
        Instantiates an Estimator object

        :param yaml_spec: A JSON object which describes the necessary model parameters and other
                          run configurations
        """
        model_fn = function_from_file(yaml_spec["model_fn_file"], "model_fn")
        run_config = RunConfig(**yaml_spec["run_config"])
        params = yaml_spec["params"]
        super(EstimatorFromYAML, self).__init__(model_fn=model_fn, params=params, config=run_config)
