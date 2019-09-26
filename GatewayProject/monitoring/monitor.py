import os
import time
import urllib
import json
import httplib
import requests

sleep = 10

def sendData():
    while True:
        state = state <- 여기에 데이터 필요
        headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
 
        try:
            # nodejs에 온도데이터를 지속적으로 요청해서 mysql에 데이터를 삽입한다
            print "[+] Node.js"
            r = requests.get('http://localhost:3000/logone', params={'state':state})
            print r.status_code
 
        except:
            print "connection failed"
        break
 
 
 
if __name__ == "__main__":
    while True:
        sendData()
        time.sleep(sleep)

