import requests_html
import requests
import os
from time import sleep
from urllib.parse import urljoin
from itertools import count

base_url = 'https://kaktusbike.sk'
base_dir = './data/training'
request_frequency = 20

session = requests_html.HTMLSession()


def parse_category(name, category_url):
    previous = None
    for i in count(0, 60):
        page = 'n,a,60,{}'.format(i)
        page_url = urljoin(base_url, category_url + '/' + page)
        print(name, page_url)
        r = session.get(page_url)
        if r.url == previous:
            break
        previous = r.url

        bikes = r.html.find('.thumb')

        # print(*[x.attrs['longdesc'] for x in bikes], sep='\n')

        for j, img_url in enumerate([x.attrs['longdesc'] for x in bikes]):
            response = requests.get(img_url, stream=True)
            if response.ok:
                fn = '/'.join((base_dir, name, str(i + j) + '.jpeg'))
                # print(fn)
                with open(fn, 'wb') as f:
                    for x in response.iter_content(1024):
                        f.write(x)
                    f.close()
            else:
                print(response)
            sleep(1 / request_frequency)

        if i > 200:
            break


def scrape():

    categories = {'road': 'cestne-bicykle',
                  # 'city': 'mestske-bicykle-skladacky',
                  'mtb': 'horske-bicykle-celoodpruzene'
                  }

    for k, v in categories.items():
        try:
            os.makedirs(base_dir + '/' + k)
        except OSError:
            pass
        parse_category(k, v)
