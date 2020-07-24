from PIL import Image
import os
import time
import constants as c
import cv2
from utility import Clogger
from utility import get_pg_conn
from psycopg2.extras import execute_values


LOG_FILENAME='preprocess.log'

logger = Clogger(LOG_FILENAME)

# TODO:
# create POSTGRES table with directory info and score values that updates
# after all files are written (and results are stored)

def setup_pg_tables(conn, cur):
    og_query = """
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};
    DROP TABLE IF EXISTS {SCHEMA}.{SOURCE_IMAGE_TABLENAME};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{SOURCE_IMAGE_TABLENAME}(
      id SERIAL PRIMARY KEY,
      full_path VARCHAR(128),
      score DOUBLE PRECISION
    );
    """.format(**c.PG_DICT)
    cur.execute(og_query)
    conn.commit()
    logger.info('Set up PG tables')
    

def main():
    # make processed subdirectory
    os.makedirs(c.PREPROCESSED_SUBDIRECTORY, exist_ok=True)
    conn = get_pg_conn()
    cur = conn.cursor()

    setup_pg_tables(conn, cur)
    
    # for each subfolder of c.ROOT_DIRECTORY, reshape the files into 256x256 from 720x720
    for subfolder in c.SUBFOLDERS:
        logger.info('On subfolder %s' % subfolder)
        folder_path =  os.path.join(
            c.ROOT_DIRECTORY,
            subfolder
        )
        files = os.listdir(folder_path)
        preprocessed_path = os.path.join(
            c.PREPROCESSED_SUBDIRECTORY,
            subfolder
        )
        os.makedirs(preprocessed_path, exist_ok=True)

        rows_to_write = []

        for fn in files:

            logger.debug('Processing %s...' % fn)
            fp = os.path.join(folder_path, fn)
            new_fn = os.path.join(
                preprocessed_path,
                fn
            )
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            scaled = cv2.resize(img, c.INPUT_DIM, cv2.INTER_AREA)
            logger.debug('Saving to %s' % new_fn)
            cv2.imwrite(new_fn, scaled)
            rows_to_write.append((new_fn, c.SCORE_MATRIX.get(subfolder)))

        if rows_to_write:
            insert_query = """
            INSERT INTO {SCHEMA}.{SOURCE_IMAGE_TABLENAME}(full_path, score)
            VALUES %s
            """.format(**c.PG_DICT)
            execute_values(
                cur,
                insert_query,
                rows_to_write
            )
            conn.commit()
            logger.info('Wrote data to database')
        else:
            logger.warn('No rows to write...')
             

            



            
    


if __name__=='__main__':
    main()
    logger.info('Done!')

