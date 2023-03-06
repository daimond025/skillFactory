import datetime
from time import sleep, time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup
import csv
from pathlib import Path


filename = "articles_info.csv"
driver_path = "/chromedriver"
base_dir = "./data"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
start_time = time()

def get_load_time(article_url, user_agent):
    #будем ждать 3 секунды, иначе выводить exception и присваивать константное значение
    try:
        # меняем значение заголовка. По умолчанию указано, что это python-код
        headers = {
            "User-Agent": user_agent
        }
        # делаем запрос по url статьи article_url
        response = requests.get(
            article_url, headers=headers, stream=True, timeout=3.000
        )
        # получаем время загрузки страницы
        load_time = response.elapsed.total_seconds()
    except Exception as e:
        print(e)
        load_time = ">3"
    return load_time

def write_to_file(output_list, filename, base_dir, firsr_run = False):
    fieldnames = ["id", "load_time", "rank", "points", "comments", "title", "url"]

    with open(Path(base_dir).joinpath(filename), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if firsr_run:
            writer.writeheader()

        for row in output_list:
            writer.writerow(row)
        csvfile.close()




def connect_to_base(browser, page_number):
    base_url = "https://news.ycombinator.com/news?p={}".format(page_number)
    for connection_attempts in range(1,4): # совершаем 3 попытки подключения
        try:
            browser.get(base_url)
            # ожидаем пока элемент table с id = 'hnmain' будет загружен на страницу
            # затем функция вернет True иначе False
            WebDriverWait(browser, 5).until(
                EC.presence_of_element_located((By.ID, "hnmain"))
            )
            return True
        except Exception as e:
            print(e)
            print("Error connecting to {}.".format(base_url))
            print("Attempt #{}.".format(connection_attempts))
    return False

def getDigitsFromStr(a):
    num_list = []

    num = ''
    for char in a:
        if char.isdigit():
            num += str(char)

    return num

def parse_html(html, user_agent):
    soup = BeautifulSoup(html, "html.parser")
    output_list = []

    # ищем в объекте soup object id, rank, score и title статьи
    tr_blocks = soup.find_all("tr", class_="athing")
    article = 0
    for tr in tr_blocks:
        article_id = tr.get("id")  # id

        article_url = ''
        conteiner_url = tr.find_all('td', class_="title")
        if len(conteiner_url) == 2 and conteiner_url[1].find('a', href=True):
            article_url = conteiner_url[1].find('a', href=True)['href']
        else:
            print('ХЕРОВО')
            exit()

        # article_url = tr.find_all("a")[1]["href"]

        # иногда статья располагается не на внешнем сайте, а на ycombinator, тогда article_url у нее не полный, а добавочный, с параметрами. Например item?id=200933. Для этих случаев будем добавлять урл до полного
        if "item?id=" in article_url or "from?site=" in article_url:
            article_url = f"https://news.ycombinator.com/{article_url}"
        load_time = get_load_time(article_url, user_agent)
        # иногда рейтинга может не быть, поэтому воспользуемся try

        # значения по умолчанию
        title = ''
        score = ''
        comments = 0
        try:
            if tr.find(class_="titlelink"):
                title = tr.find(class_="titlelink").string

            if soup.find(id=f"score_{article_id}"):
                score = soup.find(id=f"score_{article_id}").string

            if soup.find_all('a', href=f"item?id={article_id}"):
                comm = soup.find_all('a', href=f"item?id={article_id}")
                if len(comm) == 2:
                    comments = getDigitsFromStr(comm[1].string)

        except Exception as e:
            print(e)
            score = "0 points"

        article_info = {
            "id": str(article_id).strip(),
            "load_time": str(load_time).strip(),
            "rank": str(tr.span.string).strip(),
            "points": str(score).strip(),
            "comments": str(comments).strip(),
            "title": str(title).strip(),
            "url": str(article_url).strip(),
        }
        # добавляем информацию о статье в список
        output_list.append(article_info)
        article += 1
    return output_list


browser = webdriver.Chrome(executable_path='chromedriver.exe')

first_run = True
for page_number in range(10):
    print("getting page " + str(page_number) + "...")
    if connect_to_base(browser, page_number):
        sleep(5)
        output_list = parse_html(browser.page_source, user_agent)

        write_to_file(output_list, filename, base_dir, first_run)

        if first_run:
            first_run = False
    else:
        print("Error connecting to hacker news")

# завершаем работу драйвера
browser.close()
sleep(1)
browser.quit()
end_time = time()
elapsed_time = end_time - start_time
print("run time: {} seconds".format(elapsed_time))