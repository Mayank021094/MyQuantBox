import requests



URL = 'https://nsearchives.nseindia.com/corporate/xbrl/INDAS_112942_1281085_19102024093334.xml'

response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})
content = response.text

print(content)
