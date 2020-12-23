import requests
import json
import time

# check flask api
# api_url = "api/"
# port = 5000

#check torchServe api
api_url = "predictions/bert_trimsweep"
port = "8080"

url = 'http://127.0.0.1:' + port + '/' + api_url

test_texts = ["Wir hatten hier ein sehr gutes Essen.",
              "Zusammenfassung: guter Service, leckeres (höheres) Essen und schöne Lage.",
              "All dies zu einem sehr vernünftigen Preis!",
              "Das Essen war einfach schlecht.",
              "Wir hatten hier ein paar super Cocktails."]

start_time = time.time()
for text in test_texts:
    j_data = json.dumps({"text": text})
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    print(r, r.text)

print("Requests took %0.3f s" % (time.time() - start_time))