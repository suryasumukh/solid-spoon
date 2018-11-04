from collections import defaultdict
import tensorflow as tf


def average_precision_loss(y_true, y_pred):
    """
    Computes the average precision-loss of the estimator.
    This is a mesure of how far away the softmax scores are from 1.0

    :param y_true: true label tensor of shape (batch_size, nb_classes)
    :param y_pred: predicted tensor of shape (batch_size, nb_classes)

    :return: a metric operation tuple of (score tensor, update tensor)
    """
    diff = y_true - y_true * y_pred
    return tf.metrics.mean(values=diff)


def get_eval_ops(y_true, y_pred, class_map) -> defaultdict:
    """
    A function that creates Precision, Recall evaluation metric operations
    for every class in the task

    :param y_true: true label tensor of shape (batch_size, nb_classes)
    :param y_pred: predicted tensor of shape (batch_size, nb_classes)
    :param class_map: a dict (label -> index)

    :return: a dict (metric_name -> metric tensor op)
    """
    eval_metric_ops = defaultdict()
    for class_name, class_idx in class_map.items():
        b_true = tf.equal(x=tf.reshape(y_true, [-1]), y=class_idx)
        b_pred = tf.equal(x=tf.reshape(y_pred, [-1]), y=class_idx)

        precision = tf.metrics.precision(labels=b_true, predictions=b_pred)
        recall = tf.metrics.recall(labels=b_true, predictions=b_pred)
        # f1 = (2 * precision[0] * recall[0]) / (precision[0] + recall[0])

        eval_metric_ops["precision_" + class_name] = precision
        eval_metric_ops["recall_" + class_name] = recall
        # eval_metric_ops["f1_" + class_name] = (f1, tf.no_op())
    return eval_metric_ops
