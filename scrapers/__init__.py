from scrapers import kaktus
from scrapers import giant
from numpy.random import rand
from glob import glob
import shutil
import os


def clean(base_dir):
    shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    # for fn in glob(base_dir+'/*/*', recursive=True):
    #     print(fn)
    #     try:
    #         # is a file
    #         os.remove(fn)
    #     except OSError:
    #         # is a directory
    #         pass
    # exit(1)


def scrape(validation_ratio=0., base_dir='./data', rescrape=False):
    if rescrape or (not os.path.isdir(base_dir)):
        clean(base_dir)
        kaktus.scrape()
        giant.scrape()
    count = 0
    for fn in glob(base_dir+'/training/*/*.*'):
        if rand() < validation_ratio:
            fn2 = fn.replace('training', 'validation', 1).rpartition('/')[0]
            try:
                os.makedirs(fn2)
            except FileExistsError:
                pass
            shutil.move(fn, fn2)
            count += 1
    print('Validation set size: {}'.format(count))


if __name__ == '__main__':
    scrape(0.2, base_dir='../data')
