#!/usr/bin/env python
#  -*- coding:utf-8 -*-

#import urllib2
#import ssl
import json
import requests
import time
#import httplib
#from socket import socket, AF_INET, SOCK_STREAM
#import urllib
#from urllib import urlencode

ipaddr="http://10.3.7.44:5000"

def req_post():
    url = ipaddr + "/test"

    headers = \
        {
            "Content-Type": "application/json",
        }
    body = \
        {
            "test1": "F:\\face_recognition\\face-recognization\\videos\\mouth_wuyuge.mp4",
            "test2":1383838438
        }
    r = requests.post(url, headers=headers, data=json.dumps(body))
    print (r.text)

def req_get():
    url = ipaddr + "/test?test1=F:\\face_recognition\\face-recognization\\videos\\mouth_wuyuge.mp4&test2=111"
    req = requests.get(url)
    print (req.text)

'''
def getone():
    url = ipaddr + "/getone?" \
          "url=/root/face-recognization/videos/eye_wuyuge.mp4" \
          "&" \
          "size=300"
    req = requests.get(url)
    print (req.text)

def getall():
    url = ipaddr + "/getall?" \
          "url=/root/face-recognization/videos/eye_wuyuge.mp4" \
          "&" \
          "username=wuyuge" \
          "&" \
          "size=300" \
          "&" \
          "number=100"
    req = requests.get(url)
    print (req.text)

def train():
    url = ipaddr + "/train"
    req = requests.get(url)
    print (req.text)

def findfaceold():
    url = ipaddr + "/findface?" \
          "url=/root/face-recognization/cache_face/1576204814577859.jpg"
    req = requests.get(url)
    print (req.text)

def liveeye():
    url = ipaddr + "/liveeye?" \
          "url=/root/face-recognization/videos/eye_wuyuge.mp4"
    req = requests.get(url)
    print (req.text)

def livemouth():
    url = ipaddr + "/livemouth?" \
          "url=/root/face-recognization/videos/mouth_wuyuge.mp4"
    req = requests.get(url)
    print (req.text)
'''

'''
def findface():
    url = ipaddr + "/find_face?" \
          "file=/root/face-recognization/cache_face/1576204814577859.jpg"\
          "&" \
          "tolerance=0.8" \

    req = requests.get(url)
    print (req.text)
'''

def findface():
    url = ipaddr + "/find_face?" \
          "file=./videos/mouth_wuyuge.mp4"
    req = requests.get(url)
    print (req.text)



def checkface():
    url = ipaddr + "/check_face?" \
          "file=./videos/mouth_wuyuge.mp4"
    req = requests.get(url)
    print (req.text)

def learnface():
    url = ipaddr + "/learn_face?" \
          "file=./videos/mouth_wuyuge.mp4"\
          "&" \
          "username=wuyuge" \
          
          
    req = requests.get(url,timeout=600)
    #req = requests.get(url,time.sleep(6))
    print (req.text)

if __name__ == '__main__':
    #req_post()
    #req_get()

    # getone()
    # getall()
    # train()
    # findfaceold()
    # liveeye()
    # livemouth()

    findface()
    #learnface()
    #learnface()