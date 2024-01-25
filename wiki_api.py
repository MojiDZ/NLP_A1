import os
import requests as req

# WIKI API
URL = "https://en.wikipedia.org/w/api.php"

# Geography
geo_categories = [
    'Category:Biogeographic_realms',
    'Category:Biomes',
    'Category:Country',
    'Category:Ecosystems',
    'Category:Deserts',
    'Category:Landscape',
    'Category:Maps',
    'Category:Places'
]

non_geo_categories = [
    'Category:Immunology',
    'Category:Music',
    'Category:Computer science',
    'Category:Trigonometry',
    'Category:Artificial intelligence',
    'Category:Anatomy',
    'Category:Movie',
    'Category:Comedy',
]


def get_txt_from_page(pageid: str, is_geo: bool, test: bool):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "pageids": pageid,
        "formatversion": "2",
        "explaintext": 1
    }

    response = req.get(url=URL, params=params)
    data = response.json()
    # Directory creation
    text = data['query']['pages'][0]['extract']
    if not os.path.exists('./corpus'):
        try:
            os.mkdir('./corpus')

            # Directory creation for corpus training

            os.mkdir('./corpus/training')
            os.mkdir('./corpus/training/geo')
            os.mkdir('./corpus/training/non_geo')

            # Directory creation for corpus testing
            os.mkdir('./corpus/testing')
            os.mkdir('./corpus/testing/geo')
            os.mkdir('./corpus/testing/non_geo')
        except Exception as e:
            print(F"Error '{e}' while creating Directories")
    # preparing training and testing
    try:
        if test:
            if is_geo:
                with open(f'./corpus/testing/geo/{pageid}.txt', 'w') as f:
                    f.write(text)
            else:
                with open(f'./corpus/testing/non_geo/{pageid}.txt', 'w') as f:
                    f.write(text)
        else:
            if is_geo:
                with open(f'./corpus/training/geo/{pageid}.txt', 'w') as f:
                    f.write(text)
            else:
                with open(f'./corpus/training/non_geo/{pageid}.txt', 'w') as f:
                    f.write(text)

    except Exception as e:
        print(f"An error occurred while writing the file: {e}")
        # Handle the error as needed, e.g., logging, raising, or any other appropriate action


def download_corpus(wiki_category: str, is_geo: bool):
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "formatversion": "2",
        "cmtitle": wiki_category,
        "cmnamespace": "0",
        "cmtype": "page",
        "cmlimit": "100"
    }

    response = req.get(url=URL, params=params)
    data = response.json()

    total_pages = len(data["query"]["categorymembers"])
    print(f'**{wiki_category}** Total pages found: {total_pages}')

    try:
        for page in range(total_pages):
            pageid = str(data["query"]["categorymembers"][page]['pageid'])

            if page < total_pages * 0.8:
                get_txt_from_page(pageid, is_geo, False)
            else:
                get_txt_from_page(pageid, is_geo, True)
    except KeyError:
        print(f'Category ERORR! : {wiki_category}\n{data}')
        exit(-1)


def main():
    for geocat in geo_categories:
        download_corpus(geocat, True)
    for ngeocat in non_geo_categories:
        download_corpus(ngeocat, False)


if __name__ == '__main__':
    main()
