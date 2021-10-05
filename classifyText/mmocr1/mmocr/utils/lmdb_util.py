import shutil
import sys
import time
from pathlib import Path

import lmdb

# from mmocr.utils import list_from_file


def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list

def lmdb_converter(img_list_file, output, batch_size=1000, coding='utf-8'):
    # read img_list_file
    lines = list_from_file(img_list_file)

    # create lmdb database
    if Path(output).is_dir():
        while True:
            print('%s already exist, delete or not? [Y/n]' % output)
            Yn = input().strip()
            if Yn in ['Y', 'y']:
                shutil.rmtree(output)
                break
            if Yn in ['N', 'n']:
                return
    print('create database %s' % output)
    Path(output).mkdir(parents=True, exist_ok=False)
    env = lmdb.open(output, map_size=1099511627776)

    # build lmdb
    beg_time = time.strftime('%H:%M:%S')
    for beg_index in range(0, len(lines), batch_size):
        end_index = min(beg_index + batch_size, len(lines))
        sys.stdout.write('\r[%s-%s], processing [%d-%d] / %d' %
                         (beg_time, time.strftime('%H:%M:%S'), beg_index,
                          end_index, len(lines)))
        sys.stdout.flush()
        batch = [(str(index).encode(coding), lines[index].encode(coding))
                 for index in range(beg_index, end_index)]
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            cursor.putmulti(batch, dupdata=False, overwrite=True)
    sys.stdout.write('\n')
    with env.begin(write=True) as txn:
        key = 'total_number'.encode(coding)
        value = str(len(lines)).encode(coding)
        txn.put(key, value)
    print('done', flush=True)
