import importlib
import yaml


def function_from_file(file_name, fn_name):
    """
    A function that loads an external python model
    and returns a callable named fn_name if it exists
    otherwise this function raises an exception

    :param file_name: (String) name of the file/module to load
    :param fn_name: (String) name of the function to load

    :return: A callable named by fn_name
    """
    try:
        module = importlib.import_module(file_name)
        callable_fn = getattr(module, fn_name)
        assert callable(callable_fn)
    except Exception as e:
        raise e
    return callable_fn


def validate_yaml(func):
    """
    A decorator function for validated loaded YAML config

    :param func: A callable function to wrap
    :return: A wrapper function for `func`
    """
    def wrapper(*args):
        yaml_spec = func(*args)
        assert 'data' in yaml_spec
        assert 'model' in yaml_spec

        model_spec = yaml_spec["model"]
        assert 'params' in model_spec
        assert 'run_config' in model_spec
        assert 'model_fn_file' in model_spec
        return yaml_spec
    return wrapper


@validate_yaml
def load_configuration():
    """
    A function to load the configuration YAML file

    Note: This function is hard-wired to always load the
    default config - config.yaml from the root of the project

    :return: JSON representation of the YAML
    """
    try:
        with open('config.yaml') as fh:
            spec = yaml.load(fh)
    except Exception as e:
        raise e
    return spec
