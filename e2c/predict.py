# Make TF be quiet
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import tensorflow as tf
import logging
import yaml
import traceback
import random
from neo4j.exceptions import CypherSyntaxError
import zipfile
import urllib.request
import pathlib

logger = logging.getLogger(__name__)

from .model import model_fn
from .util import *
from .args import get_args
from .input import gen_input_fn
from db import *


def translate(args, question):

  checkpoint = tf.train.last_checkpoint(args['model_dir'])

  if checkpoint:
    print('restoring checkpoint %s' % checkpoint)
  else:
    print('no checkpoint found at %s' % args['model_dir'])
    exit()

  estimator = tf.estimator.Estimator(
      model_fn, model_dir=args["model_dir"], params=args)

  predictions = estimator.predict(
      input_fn=lambda: gen_input_fn(args, None, question))

  for p in predictions:
    # Only expecting one given the single line of input
    return prediction_row_to_cypher(p)


if __name__ == "__main__":

  def add_args(parser):
    parser.add_argument(
        "--graph-path", type=str, default="./data/gqa-single.yaml")
    parser.add_argument("--neo-url", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo-user", type=str, default="neo4j")
    parser.add_argument("--neo-password", type=str, default="clegr-secrets")

  args = get_args(add_args)

  logging.basicConfig()
  logger.setLevel(args["log_level"])
  logging.getLogger('e2c').setLevel(args["log_level"])

  tf.logging.set_verbosity(tf.logging.ERROR)

  while True:
    query_english = str(input("Ask a question: ")).strip()

    logger.debug("Translating...")
    query_cypher = translate(args, query_english)
    print(f"Translation into cypher: '{query_cypher}'")
    print()
