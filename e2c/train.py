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


def train(args):

  tf.logging.set_verbosity(tf.logging.DEBUG)

  session_config = tf.ConfigProto(allow_soft_placement=True)
  session_config.gpu_options.allow_growth = True
  config = tf.estiamtor.RunConfig(session_config=session_config)

  estimator = tf.estimator.Estimator(
      model_fn,
      model_dir=args["model_dir"],
      config=config,
      params=args,
      warm_start_from=args["warm_start_dir"],
  )

  eval_spec = tf.estimator.EvalSpec(input_fn=lambda: gen_input_fn(args, "eval"))

  steps_per_cycle = args["max_steps"] // args["predict_freq"]

  for i in range(args["predict_freq"]):
    max_steps = steps_per_cycle * (i + 1)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: gen_input_fn(args, "train"), max_steps=max_steps)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":

  def extend(parser):
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--tokenize-data', action='store_true')

  args = get_args(extend)

  if args["tokenize_data"]:
    expand_unknowns_and_partition(args)

  train(args)
