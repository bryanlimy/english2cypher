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

  config = tf.estimator.RunConfig(
      save_summary_steps=None, keep_checkpoint_max=3)

  estimator = tf.estimator.Estimator(
      model_fn,
      model_dir=args["model_dir"],
      params=args,
      config=config,
      warm_start_from=args["warm_start_dir"])

  eval_input_fn = lambda: gen_input_fn(args, "eval")
  train_input_fn = lambda: gen_input_fn(args, "train")

  for i in range(args['epochs']):
    estimator.train(train_input_fn)
    estimator.evaluate(eval_input_fn)


if __name__ == "__main__":

  def extend(parser):
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--tokenize-data', action='store_true')

  args = get_args(extend)

  if args["tokenize_data"]:
    expand_unknowns_and_partition(args)

  train(args)
