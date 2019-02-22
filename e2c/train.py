import tensorflow as tf
import os.path
import numpy as np
import traceback
import yaml
from collections import Counter
import logging

logger = logging.getLogger(__name__)

from .input import gen_input_fn, EOS
from .model import model_fn
from .args import get_args
from .hooks import *
from .util import *
from .build_data import *


def dump_predictions(args, predictions):
  with tf.gfile.GFile(os.path.join(args["output_dir"], "predictions.txt"),
                      "w") as file:
    for prediction in predictions:
      s = ' '.join(prediction)
      end = s.find(EOS)
      if end != -1:
        s = s[0:end]

      file.write(s + "\n")


def train(args):

  tf.logging.set_verbosity(tf.logging.DEBUG)

  max_steps = 20000
  train_steps = 2000

  config = tf.estimator.RunConfig(
      save_summary_steps=2 * args['predict_freq'], keep_checkpoint_max=3)

  estimator = tf.estimator.Estimator(
      model_fn,
      model_dir=args["model_dir"],
      params=args,
      config=config,
      warm_start_from=args["warm_start_dir"])

  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: gen_input_fn(args, "eval"))
  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: gen_input_fn(args, "train"), max_steps=train_steps)

  try:
    global_step = estimator.get_variable_value('global_step')
  except ValueError as e:
    global_step = 0

  while global_step < max_steps:
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    global_step = estimator.get_variable_value('global_step')


if __name__ == "__main__":

  def extend(parser):
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--tokenize-data', action='store_true')

  args = get_args(extend)

  if args["tokenize_data"]:
    expand_unknowns_and_partition(args)

  train(args)
