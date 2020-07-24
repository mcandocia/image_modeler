## run_model.py
# Max Candocia - maxcandocia@gmail.com
# 2020-07-20
#
# run neural network on directory of images

import argparse
import os
import cv2
import glob
import csv
import json

from network import custom_loss
from network import asymmetric_loss
from keras.utils.generic_utils import get_custom_objects
import keras
from keras.models import load_model

import constants as c
import numpy as np
from utility import Clogger
from utility import get_pg_conn
from train_model import float32

LOG_FILENAME='run_model.log'
logger = Clogger(LOG_FILENAME)



def get_options():
    parser = argparse.ArgumentParser(
        description = 'Run model on folder containing images'
    )
    parser.add_argument('input', help='Glob (pattern) used to describe input files. All must be PNG files')

    parser.add_argument(
        '--output-filename',
        default='scores.csv',
        help='Output file for scores, with column name and score column'
    )

    parser.add_argument(
        '--model-id',
        default=1,
        help='Model ID to use for loading network to evaluate with'
    )

    parser.add_argument(
        '--model-filename',
        default=None,
        required=False,
        help='If provided, will override --model-id and be used as '
        'the model rather than the one pointed to by the ID from the '
        'database'
    )

    parser.add_argument(
        '--model-tablename',
        default='{SCHEMA}.{MODEL_TABLENAME}'.format(**c.PG_DICT),
        help='Table name to load model from. Default is supplied by constants.py'
    )

    args = parser.parse_args()
    options = vars(args)
    return options


def main(options):
    files = glob.glob(options['input'])

    conn = get_pg_conn()
    cur = conn.cursor()

    get_custom_objects().update(
        {
            "custom_loss": custom_loss,
            'asymmetric_loss': asymmetric_loss
        }
    )

    if options['model_filename']:
        logger.info('Loading model')
        model = load_model(
            model_filename
        )
        # TODO: add as option
        fctr = 255.
    else:
        logger.info('Fetching path from postgres')
        fetch_query = "SELECT full_path, constants_json FROM {model_tablename} WHERE model_id={model_id}".format(**options)
        cur.execute(fetch_query)
        model_filename, constants_dict = cur.fetchall()[0]

        #constants_dict = json.loads(constants_json)
        if constants_dict['DEFAULT_DIVIDE']:
            fctr = 255.
        else:
            fctr = 1.
            
        logger.info('Loading model')
        model = keras.models.load_model(model_filename)

    results_dict = {}
    logger.info('Beginning image processing')
    for i, fn in enumerate(files):
        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)/fctr
        input_data = float32([img])
        prediction = model.predict(input_data)[0][0]
        results_dict[fn] = prediction
        if (i+1) % 10 == 0:
            logger.debug('Processed %d images' % (i+1))

    logger.info('Done processing images')

    flat_data = [
        [k, v]
        for k,v in results_dict.items()
    ]

    #print(results_dict)

    max_score = max(results_dict.values())
    min_score = min(results_dict.values())
    avg_score = np.mean(list(results_dict.values()))

    logger.debug('Min/Max/Avg: %0.3f/%0.3f/%0.3f' % (min_score, max_score, avg_score))

    logger.info('Writing data to disk...')
    
    with open(options['output_filename'], 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['full_path','score'])
        writer.writerows(flat_data)

    logger.info('Done writing file to disk.')
        






if __name__=='__main__':
    options = get_options()
    main(options)
