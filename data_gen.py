from bs4 import BeautifulSoup
import requests, sys
from multiprocessing.dummy import Pool as ThreadPool


def get_html(url_path):
    return requests.get(url_path).text


def get_links(url_path):
    data = get_html(url_path)
    soup = BeautifulSoup(data, features="html.parser")
    soup = soup.find("div", {"class": "poem"})
    links = []
    for p in soup.find_all("p"):
        for link in p.find_all("a"):
            link_str = link.get("href")
            if "=" not in link_str and "#" not in link_str:
                links.append(link_str)
    return links


def get_list(arryofarray):
    return list(set(sum(arryofarray, [])))


def poem_writter(url_path):
    try:
        file_name = (
            "_".join(url_path.replace("https://ganjoor.net/", "").split("/")) + ".txt"
        )
        data = get_html(url_path)
        soup = BeautifulSoup(data, features="html.parser")
        soup = soup.find("article")
        with open("./data/" + file_name, "w") as f:
            for bait in soup.find_all("div", {"class": "b"}):
                # sys.stdout.buffer.write(bait.encode("utf-8"))
                mesras = bait.find_all("p")
                f.write("{}  {}\n".format(mesras[0].text, mesras[1].text))
    except Exception as ex:
        print(url_path, "Not good.   ", ex)


chapters = [
    "https://ganjoor.net/saadi/mavaez/mofradat/sh2/",
    "https://ganjoor.net/saadi/divan/molhaghat/sh16/",
    "https://ganjoor.net/saadi/divan/ghazals/sh453/",
    "https://ganjoor.net/saadi/boostan/bab9/sh15/",
    "https://ganjoor.net/saadi/golestan/gbab8/sh5/",
    "https://ganjoor.net/saadi/divan/ghazals/sh596/",
    "https://ganjoor.net/saadi/mavaez/masnaviat/sh43/",
    "https://ganjoor.net/saadi/divan/molhaghat/sh25/",
    "https://ganjoor.net/saadi/divan/ghazals/sh535/",
    "https://ganjoor.net/saadi/mavaez/ghete2/sh6/",
    "https://ganjoor.net/saadi/golestan/gbab2/sh34/",
    "https://ganjoor.net/saadi/divan/molhaghat/sh27/",
    "https://ganjoor.net/saadi/golestan/gbab7/sh6/",
    "https://ganjoor.net/saadi/mavaez/robaees2/sh45/",
    "https://ganjoor.net/saadi/mavaez/ghete2/sh219/",
    "https://ganjoor.net/saadi/mavaez/mofradat/sh68/",
    "https://ganjoor.net/saadi/boostan/bab4/sh18/",
    "https://ganjoor.net/saadi/golestan/gbab3/sh20/",
    "https://ganjoor.net/saadi/divan/ghazals/sh234/",
    "https://ganjoor.net/saadi/mavaez/ghazal2/sh29/",
    "https://ganjoor.net/saadi/divan/ghazals/sh127/",
    "https://ganjoor.net/saadi/boostan/bab1/sh13/",
    "https://ganjoor.net/saadi/divan/ghazals/sh606/",
    "https://ganjoor.net/saadi/mavaez/arabi/sh12/",
    "https://ganjoor.net/saadi/mavaez/robaees2/sh30/",
]


def main():
    libs = get_links("https://ganjoor.net/saadi/")[:4]
    print("Got {} Libs".format(len(libs)))
    books = get_list([get_links(lib) for lib in libs])
    print("Got {} Books".format(len(books)))
    chapters = get_list([get_links(book) for book in books])
    print("Got {} Chapters".format(len(chapters)))

    with ThreadPool(100) as pool:
        pool.map(poem_writter, chapters)


main()
