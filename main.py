# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 08:37:42 2020

@author: wangjingxian
"""

from flask import Flask, jsonify
from flask import request
from pathlib import Path
import json
import check_face as checkface
import learn_face as learnface
import find_face as findface
import socket

socket.setdefaulttimeout(600)


app = Flask(__name__)

@app.route('/check_face', methods=['GET'])
def check_face():
    if request.method == "GET":
        url = request.args.get("file")
        if checkurl(url) == False:
            js = {"status": 0}
            return json.dumps(js)
        status,image = checkface.cap_one_image(url)
        js = {"status": status,"image": image }
        return json.dumps(js)


@app.route('/learn_face', methods=['GET'])
def learn_face():
    if request.method == "GET":
        url = request.args.get("file")
        if checkurl(url) == False:
            js = {"status": 0}
            return json.dumps(js)
        username = request.args.get("username")
      
        status = learnface.dataset_construction(url,username)
        
        if status==1:
            #print('666666666666666666666666666')
            status = learnface.dataset_train()
            #print('7777777777777777777777777')
        js = {"status": status}
        return json.dumps(js)


@app.route('/find_face', methods=['GET'])
def find_face():
    if request.method == "GET":
        url = request.args.get("file")
        if checkurl(url) == False:
            js = {"status": 0}
            #print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            return json.dumps(js)
        status1 = findface.live_detection_eye(url)
        #print('bbbbbbbbbbbbbbbbbbbb')
        status2 = findface.live_detection_mouth(url)
        
        username=""
        if status1==1 or status2==1:
            
            status,image=checkface.cap_one_image(url)
            
            if status==1:
                status,username = findface.get_faces(image)
            else:
                status=0
        else:
            status=0
        js = {"status": status, "user_name": username}
        return json.dumps(js)




def checkurl(url):
    my_file = Path(url)
    if my_file.exists():
        return True
    else:
        return False

if __name__ == '__main__':
    app.run(debug=True,port=5000,host="0.0.0.0")