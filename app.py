from flask import Flask, render_template, request, url_for, session, redirect
import numpy as np
import pandas as pd
import os
import time
import glob
import pathlib
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
from nltk.tokenize import sent_tokenize
# import google.colab
# from google.colab import drive
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import IPython.display as display
import moviepy.editor as mp
import sklearn
import subprocess as sp
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import bcrypt


app = Flask(__name__)

#Function for Load the lecture video (video.mp4)
def loadvideo(LECTURE_PATH):
  lectvid = mp.VideoFileClip(LECTURE_PATH)
  lectvid.audio.write_audiofile(r"lectureaudio.wav")
  return librosa.load("lectureaudio.wav", sr = 16000)#returns (audio, rate)

# Import model & tokenizer
tokenizertrans = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#a function to do the transcription
def transcriptor(audio):
  #tokenize the audio
  input_values = tokenizertrans(audio, return_tensors = "pt").input_values

  #get the logit values
  logits = model(input_values).logits

  #get predicted Values
  prediction = torch.argmax(logits, dim = -1)

  #decode the predicted values and return
  return tokenizertrans.batch_decode(prediction)[0]

#transcription splitter. This is because if the audio is too long, then the model will eat too much ram
#audio = audio variable, print_step = print each transcription or not, SPLITLEN = split audio by x seconds
#bigger splitlen is better but more ram expensive. Try to find the maximum splitlen you can use
def transcript(audio, rate, print_step = False, SPLITLEN = 10):
  #define the output variable
  out = ''

  #loop according to audio length and splitlen
  for a in range(len(audio)//SPLITLEN):
    #make a variable to store the split audio
    y = audio[SPLITLEN * rate * a: SPLITLEN * rate *(a+1)]
    #print(y)
    #if y is not empty array
    if y.size > 0:

      #print each step of the transcription process or not
      if print_step:
        print(out)

      #append y to output
      out += ' ' + transcriptor(y)

  return out

#load tokenizer for summarization
checkpoint = "t5-small"
tokenizersum = AutoTokenizer.from_pretrained(checkpoint)

#load transcription summarizer
modelpath = './NMC'

summarizer = pipeline("summarization", model=modelpath)

#splitter function
#here's an important function:
#this is the function used to get around BERT's 512 token limitation
#this one splits the (text) into X amount of (Splitlen) token long sentences, summarizes those then concantenates the results.
def summarizetext(text, splitlen):
    #tokenize the text
    token = tokenizersum(text)
    print(len(token['attention_mask']))
    #load an empty tokenized text
    splittoken = tokenizersum('')

    #load an empty variable for output
    outtext = ''

    #loop based on the length of the tokenized text divided by the split length
    for i in range(len(token['attention_mask'])//splitlen):

      #splits the token for both the input ids and the attention mask by the split length. The output is kept in splittoken
      splittoken['input_ids'] = token['input_ids'][i*splitlen:(i+1)*splitlen:]
      splittoken['attention_mask'] = token['attention_mask'][i*splitlen:(i+1)*splitlen:]

      #Decode the resulting split token and summarize a chunk of the text using the model. Put the result in output
      output = (summarizer(tokenizersum.decode(splittoken["input_ids"], max_length = 512)))

      #concatenate the resulting output into a variable.
      outtext += output[0]["summary_text"]

      #return the result
    return outtext

# connect to database
app.secret_key = 'xyzsdfg'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user-system'

mysql = MySQL(app)

@app.route('/', methods = ['GET'])
def index():
	return render_template('main.html')

@app.route('/team')
def team():
  return render_template('transcript.html')

@app.route('/main')
def main():
  return render_template('main.html')

@app.route('/main2')
def main2():
  return render_template('main2.html')

@app.route('/subscription')
def subscription():
  return render_template('subscription.html')

# @app.route('/login')
# def login():
#   return render_template('login.html')

# @app.route('/signup')
# def signup():
#   return render_template('signup.html')
@app.route('/account')
def account():
  return render_template('account.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/about2')
def about2():
  return render_template('about2.html')

@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        password = password.encode('utf-8')
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        user = cursor.fetchone()
        userPassword = user['password'].encode('utf-8')
        verify = bcrypt.checkpw(password, userPassword)
        if user and verify == True:
            session['loggedin'] = True
            session['name'] = user['name']
            session['userid'] = user['userid']
            session['email'] = user['email']
            session['plan'] = user['plan']
            mesage = 'Logged in successfully !'
            return render_template('main2.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)

@app.route('/', methods = ['POST'])
def predict():
    message = ''
    videofile = request.files['videofile']      

    videofile.save("./data.mp4")
        
    LECTURE_PATH = "./data.mp4"

    size = os.stat(LECTURE_PATH).st_size

    if session['plan'] == 'free' and size > 52428800:
      message = "The file is too big, you need file smaller than 50 MB"
      return render_template('transcript.html', message = message)

    audio, rate = loadvideo(LECTURE_PATH)

    transcription = transcript(audio, rate, print_step = True, SPLITLEN = 30)
    
    summary = summarizetext(transcription.lower(),128)

    with open('./static/lecturesummary.txt', 'w') as f:
        f.write(summary.lower())

    message = "Please download the .txt file from here:"
	
    return render_template('transcript-download.html', prediction = message)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    session.pop('plan', None)
    return redirect(url_for('login'))
  
@app.route('/signup', methods =['GET', 'POST'])
def signup():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        bytePwd = password.encode('utf-8')
        mySalt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(bytePwd, mySalt)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        elif len(password) < 8:
            mesage = 'Password less than 8 characters'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s,% s, % s, % s, NULL)', (email, password_hash, userName, "free"))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('signup.html', mesage = mesage)

@app.route('/signup2', methods =['GET', 'POST'])
def signup2():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'card' in request.form:
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        card = request.form['card']
        bytePwd = password.encode('utf-8')
        mySalt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(bytePwd, mySalt)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email or not card:
            mesage = 'Please fill out the form !'
        elif len(password) < 8:
            mesage = 'Password less than 8 characters'
        elif card.isnumeric() == False:
            mesage = 'The card number must be number'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s,% s, % s, % s, % s)', (email, password_hash, userName, "paid", card, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('signup2.html', mesage = mesage)


if __name__ == "__main__":
	app.run(debug=True)