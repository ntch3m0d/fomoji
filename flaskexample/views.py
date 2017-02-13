# -*- coding: utf-8 -*-
import os
from flask import Flask,request,Response,render_template,url_for, send_from_directory
from flaskexample import app
from werkzeug import secure_filename
import hashlib
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFont, ImageFilter
from PIL import ImageDraw
import textwrap
import emoji
import sys
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2
global shortname
global current_emoji

# ubuntu paths
# app.config['RAW_IMAGE_FOLDER']='/home/ubuntu/application/tmp' #/Users/NVT/PycharmProjects/application2/flaskexample/static/images/RAW/'
# app.config['MODELS_FOLDER']='/home/ubuntu/application/flaskexample/models/'
# app.config['STATIC_FOLDER']='/home/ubuntu/application/flaskexample/static'

# osX paths
app.config['RAW_IMAGE_FOLDER']='/Users/NVT/fomoji/tmp' #/Users/NVT/PycharmProjects/application2/flaskexample/static/images/RAW/'
app.config['MODELS_FOLDER']='/Users/NVT/fomoji/flaskexample/models/'
app.config['STATIC_FOLDER']='/Users/NVT/fomoji/flaskexample/static'

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif','JPG'])
app.config['EMOJIS']=["wink", "facepalm", "yell", "kiss", "smile"] # ORDER MATTERS!!!!!!

#app.config['HAARmodel']=cv2.CascadeClassifier("/Users/NVT/PycharmProjects/application2/flaskexample/models/haarcascade_frontalface_default.xml")
#app.config['LDAmodel']=
#vc = cv2.VideoCapture(0)
fisherfaces = cv2.createFisherFaceRecognizer(0)

#ubuntu paths
# facebox = cv2.CascadeClassifier("/home/ubuntu/application/flaskexample/models/haarcascade_frontalface_default.xml")
# #fisherfaces.load('/home/ubuntu/application/flaskexample/models/allfellows_detection_model.xml')
# fisherfaces.load('/home/ubuntu/application/flaskexample/models/fellows_and_00_detection_model.xml')

# osX paths:
facebox = cv2.CascadeClassifier("/Users/NVT/fomoji/flaskexample/models/haarcascade_frontalface_default.xml")
# # fisherfaces.load('/Users/NVT/PycharmProjects/INSIGHT_APP/fomoji/models/allfellows_detection_model.xml')
fisherfaces.load('/Users/NVT/fomoji/flaskexample/models/fellows_and_00_detection_model.xml')


@app.route('/')
@app.route('/index')
def index():
    return render_template('frontpage.html')

@app.route('/upload', methods=['POST'])
def upload():
    global shortname
    global current_emoji
    current_emoji="none"
    print("HELLO!")
    myfile=request.files['file']
    # myfile2=request.args.get['file']

    filename = secure_filename(myfile.filename)
    print(filename)
    fileext = filename.rsplit('.', 1)[1] # split after .
    timestr = time.strftime("%Y%m%d-%H%M%S")
    m = hashlib.md5()
    m.update(filename + timestr)
    filehash = m.hexdigest()
    tmpfilename = os.path.join(app.config['RAW_IMAGE_FOLDER'], filehash + "." + fileext)
    shortname=filehash + "." + fileext
    myfile.save(tmpfilename)     #save

    # load the pic and resize. Save resized.
    img=cv2.imread(tmpfilename)
    w2=640
    small_frame=resize_pic(img,w2)  # resize image to max width of 640 pxls
    #print("TOUCHING ", filename, " and ", fileext,"and",tmpfilename)

    # this function:
    # 1)converts pic to grey and finds face box w/ haar cascade.
    # 2)plots blue box around face on non-grey pic and returns as small_frame
    # 3)grey face box is sent to decoder to be classified as an emoji.
    # emoji classification,confidence, and blue-boxed colored image are returned
    current_emoji, conf,small_frame = detect_face_box(small_frame, facebox)
    cv2.imwrite(os.path.join(app.config['RAW_IMAGE_FOLDER'], filehash +"." + fileext), small_frame)



    return render_template('image_loaded.html',
                           bounded_filename = url_for('uploaded_file', filename=shortname))

@app.route('/populate', methods=['POST','GET'])
def populate():
    # -*- coding: utf-8 -*-
    global shortname
    global current_emoji
    # here, we import text in the text box, and populate our default bubble with it.
    select = request.form.get('comp_select')
    mytext=str(select)
    #return(str(select)) # just to see what select is
    img = Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/SpeechBubble_empty.jpg"))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.join(app.config['STATIC_FOLDER'],"fonts/Arial Unicode.ttf"), 72,
                               encoding='unic')
    emofont=ImageFont.truetype(os.path.join(app.config['STATIC_FOLDER'],"fonts/NotoEmoji-Regular.ttf"), 72,
                               encoding='unic')
    margin = offset = 70
    if current_emoji=="smile":
        emoji_image=Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/smile_open.png"))
    elif current_emoji== "facepalm":
        emoji_image=Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/facepalm_google.png"))
    elif current_emoji=="kiss":
        emoji_image=Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/kiss_open.png"))
    elif current_emoji=="yell":
        emoji_image=Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/yell_open.png"))
    elif current_emoji=="wink":
        emoji_image=Image.open(os.path.join(app.config['STATIC_FOLDER'],"images/wink_open.png"))
    else:
        emoji_image=[]


    if len(mytext)>1:
        for line in textwrap.wrap(mytext, width=30):
            draw.text((margin, offset), line, font=font, fill="#aa0000")
            offset += font.getsize(line)[1]
    else:
        draw.text((margin, offset), '', font=font, fill="#aa0000")

    emoji_image.thumbnail((100, 100), Image.ANTIALIAS)  # resizes image in-place
    background = Image.new('RGBA', img.size, (255, 255, 255, 0))

    xoff = 100
    try:
        img.paste(emoji_image, (xoff, offset), mask=emoji_image)
    except ValueError:
        pass

    img = img.convert('RGBA')
    background.paste(img, img)

    #here we want to save the bubble as the same name as the image it came from (but with word 'bubble' appended)
    bubble_name=shortname.rsplit('.', 1)[0] # split before .
    bubble_ext=shortname.rsplit('.', 1)[1] # split after .
    bubblewithtextname=bubble_name+"_bubble."+bubble_ext

    #now add emoji to corner!!!
    background.save(os.path.join(app.config['RAW_IMAGE_FOLDER'], bubblewithtextname),"JPEG")
    print(bubblewithtextname)
    #return(current_emoji)
    # bubblewithtextname="sample-out.jpg"
    return render_template('BubblePopulated.html',
                           bounded_filename = url_for('uploaded_file', filename=shortname),
                           text_and_bubble = url_for('uploaded_file', filename=bubblewithtextname))


def resize_pic(img,w2):
    h_w = img.shape # get old shape
    h = h_w[0] # height
    w = h_w[1] # width
    h2 = h * w2 / w #new height w/ same aspect ratio
    newimg = cv2.resize(img, (w2, h2)) #remake
    print("RESIZED TO:", str(newimg.shape))

    return newimg

def detect_face_box(small_frame,facebox):
    current_emoji="none"
    conf=0
    small_grey_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)  #grey it
    face = facebox.detectMultiScale(small_grey_frame, scaleFactor=1.1, minNeighbors=1, minSize=(150, 150),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) > 0:
        #draw rectangle
        for (x, y, w, h) in face:
            print("FOUND FACE!")
            cv2.rectangle(small_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #draw rectangle and save
            #NOTE: for some reason this thing detects boxes that are much wider than in training set. the 10 is to compensate...ugh
            #current_emoji,conf=classify_face(fisherfaces, small_grey_frame[y+10:y+h-10,x+10:x+w-10])
            current_emoji,conf=classify_face(fisherfaces, small_grey_frame[y:y+h,x:x+w])

            #current_emoji is a string...
            print(current_emoji,conf)
    return current_emoji,conf,small_frame


def classify_face(fisherfaces,small_grey_frame):
    emojis=["wink", "facepalm", "yell", "kiss", "smile", "none"]
    training_size_grey_frame = cv2.resize(small_grey_frame, (350, 350))
   # print(training_size_grey_frame.shape[:2])
    cv2.imwrite("webcam.jpg",training_size_grey_frame)
    pred, conf = fisherfaces.predict(training_size_grey_frame)
    if pred==-1:
        detected_emoji="none"
    else:
        detected_emoji=emojis[pred]
    return detected_emoji,conf

@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RAW_IMAGE_FOLDER'],filename)

@app.route('/project_info')
def project_info():
    return render_template('project_info.html')


