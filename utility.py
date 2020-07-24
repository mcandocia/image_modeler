import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import logging

import dbinfo
import psycopg2
import random

#used for processing these intervals
def urand(tup):
    return random.uniform(tup[0],tup[1])


class CStreamHandler(logging.StreamHandler):
    
    def format(self, record):
        msg = super().format(record)
        color_prefix = {
            10: '0',
            20: '1;36',
            30: '1;33',
            40: '1;31',
            50: '1;37;41',
        }[record.levelno]

        msg = '\x1b[%sm%s\x1b[0m' % (color_prefix, msg)
        
        return msg
        
    


class Clogger(object):
    idx = 0
    def __init__(
            self,
            filename='default.log'
        ):
        Clogger.idx+=1
        self.filename=filename

        self.logger = logging.getLogger('logger_%d' % Clogger.idx)
        self.logger.setLevel(logging.DEBUG)
        sh = CStreamHandler()
        fh = logging.FileHandler(filename)
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            '%Y-%m-%d %H:%M:%S',
        )

        sh.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        

    def __getattr__(self, *args, **kwargs):
        return getattr(self.logger, *args, **kwargs)

def get_pg_conn():
    conn = psycopg2.connect(
        "dbname=%s user=%s password=%s host=%s port=%s" %
        ( dbinfo.dbname, dbinfo.username, dbinfo.pw, dbinfo.host, dbinfo.port)
    )
    return conn
