## train_model.py
# Max Candocia - maxcandocia@gmail.com
# 2020-07-18
#
# train the model

import cv2
from utility import Clogger
import constants as c
from utility import get_pg_conn
from network import Network
from network import custom_loss
from network import asymmetric_loss
from network import asymmetric_loss2
import json
from datetime import datetime
import os
import numpy as np
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import keras
from keras.models import load_model

import sys

get_custom_objects().update(
        {
            "custom_loss": custom_loss,
            'asymmetric_loss': asymmetric_loss,
            'asymmetric_loss2': asymmetric_loss2,
        }
)

LOG_FILENAME='train_model.log'
logger=Clogger(LOG_FILENAME)

def prepare_for_json(x, tuple_to_string=False):
    if isinstance(x, dict):
        x = {
            prepare_for_json(k, True): prepare_for_json(v)
            for k, v in x.items()
        }
    elif isinstance(x, (list, tuple)):
        if tuple_to_string and isinstance(x, tuple):
            x = str(x)
        else:
            x = [
                prepare_for_json(v)
                for v in x
            ]
    elif isinstance(x, (str, int, float)):
        return x
    elif isinstance(x, (set, complex)):
        x = str(x)
    elif x is None:
        pass
    else:
        x = repr(x)

    return x
        
    

def module_to_json(m):
    varnames = [
        v for v in dir(m)
        if not v.startswith('__')
    ]

    return json.dumps(prepare_for_json({
        v: getattr(m, v)
        for v in varnames
    }))
    


def float32(x):
    return np.float32(np.asarray(x))

if c.DEFAULT_DIVIDE:
    divisor = 255.
else:
    divisor = 1.

def load_fetched_images(results, divide=c.DEFAULT_DIVIDE):
    #logger.debug('Loading fetched data')
    # allows for unscored results to use this function
    if isinstance(results[0], str):
        results = [(r, None) for r in results]
    if divide:
        fctr = 255
    else:
        fctr = 1
    xlist = [cv2.imread(r[0], cv2.IMREAD_UNCHANGED)/fctr for r in results]
    ylist = [r[1] for r in results]
    return float32(xlist), float32(ylist)
    

class TrainFetcher:
    modulo_base=12

    # train, test, validation
    modulo_ranges = [
        0,
        7,
        9,
        11,
    ]
        
    
    fetch_query_base = """
    SELECT d.full_path, s.score FROM
    {SCHEMA}.{SOURCE_IMAGE_TABLENAME} s 
    INNER JOIN
    {SCHEMA}.{DISTORTION_IMAGE_TABLENAME} d
    ON s.id=d.id %(condition)s
    ORDER BY md5(d.full_path) -- pre-determined, random
    """.format(**c.PG_DICT)
    
    def __init__(
            self,
            batch_size=c.DEFAULT_TRAIN_BATCH_SIZE
    ):
        self.batch_size = batch_size
        self.train_conn = get_pg_conn()
        self.validation_conn = get_pg_conn()
        self.test_conn = get_pg_conn()
        
        self.train_cur = self.train_conn.cursor()
        self.validation_cur = self.validation_conn.cursor()
        self.test_cur = self.test_conn.cursor()

        self.initialize_queries()

    def initialize_queries(self):
        # train, test, validation
        queries = [
            self.__class__.fetch_query_base %
            {'condition': ' AND d.distort_id %% %s BETWEEN %s AND %s' %
             (
                 self.__class__.modulo_base,
                 self.__class__.modulo_ranges[i],
                 self.__class__.modulo_ranges[i+1],
             )
            } for i in range(3)
        ]
        
        self.queries = {
            ['train','validation','test'][i]: queries[i]
            for i in range(3)
        }

        self.train_cur.execute(self.queries['train'])
        self.validation_cur.execute(self.queries['validation'])
        self.test_cur.execute(self.queries['test'])

    def fetch_train(self, n=None):
        if n is None:
            target = self.batch_size
        else:
            target = n
        #logger.debug('Fetching training data from SQL')
        res = self.train_cur.fetchmany(target)            
        num_results = len(res)
        if num_results < self.batch_size:
            self.train_cur.execute(self.queries['train'])
            res += self.train_cur.fetchmany(target - num_results)

        #logger.warn(res[0])
        return res

    def fetch_validation(self, n=None):
        if n is None:
            target = self.batch_size
        else:
            target = n
        res = self.validation_cur.fetchmany(target)            
        num_results = len(res)
        if num_results < self.batch_size:
            self.validation_cur.execute(self.queries['validation'])
            res += self.validation_cur.fetchmany(target - num_results)
        return res

    def fetch_test(self, n=None):
        if n is None:
            target = self.batch_size
        else:
            target = n
        res = self.test_cur.fetchmany(target)
        logger.debug('Fetching testing data from SQL')        
        num_results = len(res)
        if num_results < self.batch_size:
            self.test_cur.execute(self.queries['test'])
            res += self.test_cur.fetchmany(target - num_results)
        return res    


def set_up_model_table(conn, cur):
    create_query = """
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{MODEL_TABLENAME}(
      model_id SERIAL,
      full_path VARCHAR(128),
      final_mse DOUBLE PRECISION,
      final_mae DOUBLE PRECISION, 
      training_perf_history JSON,
      constants_json JSON
    )
    """.format(**c.PG_DICT)
    logger.info('Setting up model table')
    cur.execute(create_query)
    conn.commit()
    logger.info('Model table set up')


def main():
    # for recordkeeping
    constants_json = module_to_json(c)
    logger.debug('Constants JSON: %s' % constants_json)
    
    fetcher = TrainFetcher()
    conn = get_pg_conn()
    cur=conn.cursor()

    set_up_model_table(conn, cur)

    os.makedirs(c.MODEL_DIRECTORY, exist_ok=True)

    if len(sys.argv) == 1:
        logger.info('Creating new network')
        model = Network(loss=asymmetric_loss2)
        logger.info('Built Network')
    else:
        logger.info('Loading model with path %s' % sys.argv[1])
        model = load_model(sys.argv[1])
        logger.info('Loaded network')

    logger.info('Beginning training iterations')
    test_scores = []
    try:
        for epoch in range(c.N_TRAINING_EPOCHS):
            logger.info('Starting epoch #%s' % epoch)
            for i in range(c.N_BATCHES_PER_EPOCH):
                if (i+1) % 10 == 0:
                    logger.debug('%d batches in this epoch' % (i+1))

                x_train, y_train = load_fetched_images(fetcher.fetch_train())
                model.train_on_batch(x_train, y_train)

            logger.info('End of training for epoch. Testing...')
            x_test, y_test = load_fetched_images(
                fetcher.fetch_test(c.TEST_SAMPLE_SIZE)
            )
            test_eva = model.evaluate(x_test, y_test)
            logger.debug(test_eva)
            test_scores.append(test_eva)
    except KeyboardInterrupt:
        logger.warn('KeyboardInterrupt detected. Ending training')

    logger.info('Saving model')
    model_ts_sfx = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model_fn = 'model_%s.h5' % model_ts_sfx
    model_path = os.path.join(
        c.MODEL_DIRECTORY,
        model_fn
    )
    model.save(model_path)

    insert_query = """ 
    INSERT INTO {SCHEMA}.{MODEL_TABLENAME}(full_path, final_mse, final_mae, training_perf_history, constants_json) VALUES (%s, %s, %s, %s, %s)
    """.format(**c.PG_DICT)
    cur.execute(insert_query, [model_path, test_scores[-1][1], test_scores[-1][2], json.dumps(test_scores),constants_json])
    conn.commit()
    conn.close()
            
    
    
if __name__=='__main__':
    main()
    logger.info("DONE!")
    
    
