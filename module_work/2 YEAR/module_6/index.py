import requests
from bs4 import BeautifulSoup

# full url = "https://en.soccerwiki.org/league.php?leagueid=78"
base_url = "https://en.soccerwiki.org/"
add_url = "league.php?leagueid=78"

response = requests.get(base_url + add_url).text
soup = BeautifulSoup(response, 'html.parser')

div = soup.find_all('div', class_='table-custom-responsive mb-3')[0]
table = div.find('table', attrs={'class': 'table-custom table-roster'})
rows = table.select('td.text-left:nth-child(2)')
clubs = []
for td in rows:
    try:

        clubs.append(td.find('a').attrs['href'])
    except Exception as e:
        print(e)

all_age = []
for club in clubs:
    html_club = requests.get(base_url + club).text
    soup_club = BeautifulSoup(html_club, 'html.parser')

    table = soup_club.find_all('table', attrs={'class': 'table-custom table-roster'})[0]
    td_age = table.select('td:nth-child(6)')
    for age in td_age:
        all_age.append(int(age.get_text()))


print(round(sum(all_age)/len(all_age)))
exit()
