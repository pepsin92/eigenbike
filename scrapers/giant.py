import requests_html
import requests
import os
from time import sleep
from urllib.parse import urljoin
from itertools import count

base_url = 'https://www.giant-bicycles.com/int/bikes'
base_dir = './data/training'
request_frequency = 20

session = requests_html.HTMLSession()


def parse_category(name, category_url, name_prefix=''):

    page_url = base_url + '/' + category_url
    print(name, page_url)
    r = session.get(page_url)

    bikes = r.html.find('.img-responsive')[2:]

    # print(*[x.attrs['src'] for x in bikes], sep='\n')
    # exit(0)
    for j, img_url in enumerate([x.attrs['src'] for x in bikes]):
        response = requests.get(img_url, stream=True)
        if response.ok:
            fn = '/'.join((base_dir, name, 'giant_'+name_prefix+str(j) + '.jpeg'))
            with open(fn, 'wb') as f:
                for x in response.iter_content(1024):
                    f.write(x)
        else:
            print(response)
        sleep(1 / request_frequency)


def scrape():
    categories = [['road', 'on-road/aero-race', 'aero-race'],
                  ['road', 'on-road/race', 'race'],
                  ['road', 'on-road/endurance', 'endurance'],
                  ['road', 'on-road/triathlon-tt', 'tt'],
                  ['road', 'on-road/all-rounder', 'all-rounder'],
                  ['road', 'x-road/cyclocross', 'cyclocross'],
                  # ['city', 'on-road/city', 'city'],
                  ['mtb',  'off-road/race-xc', 'race-xc'],
                  ['mtb',  'off-road/xc', 'xc'],
                  ['mtb',  'off-road/trail', 'trail'],
                  ['mtb',  'off-road/gravity', 'gravity']
                  ]

    for k in categories:
        try:
            os.makedirs(base_dir + '/' + k[0])
        except OSError:
            pass
        parse_category(k[0], k[1], k[2])
