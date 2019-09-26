import binascii
import struct
import time

import urllib
import json
import httplib
import requests

from bluepy.btle import UUID, Peripheral

temp_uuid = UUID(0x180A)

p = Peripheral("E0:95:58:90:DD:9D", "random")
prev = ''
try:
    svc = p.getServiceByUUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
    ch = svc.getCharacteristics("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")[0]
    print(ch.read())
    prev = ''
    state = 0
    while 1:
        try:
            data = str(ch.read()).strip()
            if data=='st1' or data == 'st2':
                print("stop")
                state = 0
                prev = 'stop'
            elif data == "wlk":
                print("walk")
                state = 1
                prev = 'walk'
            elif data == "sit":
                print("sit or stand")
                state = 2
                prev = 'sit or stand'
            elif data == "run":
                print("run")
                state = 3
                prev = 'run'
            elif data == "fal":
                print("fall")
                state = 4
                prev = 'fall'
            elif data == 'st3':
                print(prev)
            
            headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
     
            try:
                print "[+] Node.js"
                print (state)
                r = requests.get('http://localhost:3000/logone', params={'state':state})
                print r
                r = requests.get('http://localhost:3000/graph', params={'state':state})
                print r
     
            except:
                print " "
            time.sleep(1.5)
        except Exception as e:
            try:
                p.disconnect()
            except Exception as e2:
                print(e2)
            p = Peripheral("E0:95:58:90:DD:9D", "random")
            svc = p.getServiceByUUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
            ch = svc.getCharacteristics("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")[0]            
finally:
    p.disconnect()
