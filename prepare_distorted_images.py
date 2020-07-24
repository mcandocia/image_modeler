## prepare_distorted_images.py
# Max Candocia - Max.Candocia@gmail.com
# 2020-07-18
#
# prepare all transformations/distortions of input data in advance,
# as to avoid expensive computation while training

from distortion_wheel import DistortionWheel
import constants as c
from utility import get_pg_conn
from utility import Clogger
from copy import deepcopy
import random
import sys
import os
import shutil
from uuid import uuid4
import re
from psycopg2.extras import execute_values
import cv2

LOG_FILENAME='prepare_distorted_images.log'

# number of unique distortions
N_DISTORTION_ITERATIONS=c.N_DISTORTION_ITERATIONS

# percentage of images to use for each distortion
DISTORTION_SAMPLE_RATE=c.DISTORTION_SAMPLE_RATE

logger = Clogger(LOG_FILENAME)

def setup_distortion_table(conn, cur):
    os.makedirs(c.DISTORTION_FOLDER, exist_ok=True)

    if len(sys.argv) == 2:
        if sys.argv[1] == 'reset':
            logger.warn(
                'Resetting table and clearing out previous '
                'distorted images.'
            )
            drop_query = """
            DROP TABLE IF EXISTS
            {SCHEMA}.{DISTORTION_IMAGE_TABLENAME}
            """.format(**c.PG_DICT)
            logger.info('Executing query')
            cur.execute(drop_query)
            conn.commit()
            logger.info('Deleting directory')
            shutil.rmtree(c.DISTORTION_FOLDER)
            logger.info('Recreating directory')
            os.makedirs(c.DISTORTION_FOLDER)
        else:
            logger.warn('Unknown argument. Only "reset" understood. Ignoring.')

    create_query = """
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{DISTORTION_IMAGE_TABLENAME}(
      distort_id SERIAL PRIMARY KEY,
      id INTEGER,
      full_path VARCHAR(128),
      source_full_path VARCHAR(128),
      iteration INTEGER
    )
    """.format(**c.PG_DICT)
    cur.execute(create_query)
    conn.commit()
            
    logger.info("Set up distortion image table")

def get_filename_list(cur):
    fetch_query = """
    SELECT id, full_path FROM {SCHEMA}.{SOURCE_IMAGE_TABLENAME}
    """.format(**c.PG_DICT)
    cur.execute(fetch_query)
    
    return({
        x[0]: x[1]
        for x in cur.fetchall()
    })

def load_files(filenames_dict):
    logger.info('Loading image files')
    images = {
        id: cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        for id, fn in filenames_dict.items()
    }
    logger.info('Loaded %s files' % len(images))
    return images
        

    

def main():
    logger.info("Setting connection up")
    # load postgres db
    conn = get_pg_conn()
    cur = conn.cursor()

    setup_distortion_table(conn, cur)


    # get filename list
    filenames_dict = get_filename_list(cur)

    # load all files (memory should be enough)
    data = load_files(filenames_dict)

    sample_size = round(DISTORTION_SAMPLE_RATE * len(data))
    logger.debug('Sample size: %s' % len(data))
    logger.debug('Distortion sample size: %s' % sample_size)
    wheel = DistortionWheel(*c.INPUT_DIM, use_color=False, debug=True)

    max_iter_query  = """
    SELECT max(iteration) FROM {SCHEMA}.{DISTORTION_IMAGE_TABLENAME}
    """.format(**c.PG_DICT)
    cur.execute(max_iter_query)
    max_iter = cur.fetchall()[0][0]
    if max_iter is None:
        max_iter = -1
    # make copies, run through distorter, and save to disk/postgres
    for i in range(max_iter+1, max_iter + 1 + N_DISTORTION_ITERATIONS):
        logger.info('On iteration %s of distortions' % i)
        logger.debug('Selecting sample')
        image_sample_dict ={
            id: data[id]
            for id in 
            random.sample(
                list(data),
                sample_size
            )
        }
        logger.debug('Sampled images')
        
        if i != 0:
            logger.debug('Adjusting distorter')
            wheel.initialize_distortions()


        row_data = []
        logger.debug('Iterating through images')
        cnt = 0
        for id, image in image_sample_dict.items():
            
            # to avoid modifying original
            image_copy = deepcopy(image)

            if True:
                processed_image = wheel.process_image(image_copy, smart=False)
            else:
                processed_image = image_copy
                
            source_fn = filenames_dict[id]
            
            source_path, non_path_filename = os.path.split(source_fn)
            src_fldr=os.path.split(source_path)[1]
            random_id = str(uuid4())
            os.makedirs(
                os.path.join(
                    c.DISTORTION_FOLDER,
                    src_fldr
                ),
                exist_ok=True
            )

            new_fn = os.path.join(
                c.DISTORTION_FOLDER,
                src_fldr,
                '%s_%s.png' % (non_path_filename[:-4], random_id)
            )

            # write output
            cv2.imwrite(new_fn, processed_image)

            row_data.append(
                (
                    id,
                    new_fn,
                    source_fn,
                    i
                )
            )
            cnt+=1
            if cnt % 25 == 0:
                logger.debug('distorted %s images in this iteration' % cnt)

        logger.info('Done with iteration. Writing data to postgres')
        write_query = """
        INSERT INTO {SCHEMA}.{DISTORTION_IMAGE_TABLENAME}(
          id,
          full_path,
          source_full_path,
          iteration
        ) VALUES %s

        """.format(**c.PG_DICT)
        execute_values(cur, write_query, row_data)
        conn.commit()

        logger.info('Done writing to postgres')
        

if __name__=='__main__':
    main()
    logger.info('Done with all iterations!')

