# extract pure corpus from OANC dataset

import os
import sys
import numpy as np
import codecs
import re
import random

# >>>f = codecs.open("test", "r", "utf-8")
sourcedir = 'D:\OANC-1.0.1-UTF8\OANC\data'

walk_dir = sourcedir
# walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)

# from http://stackoverflow.com/a/2212698

# If your current working directory may change during script execution, it's recommended to
# immediately convert program arguments to an absolute path. Then the variable root below will
# be an absolute path as well. Example:
# walk_dir = os.path.abspath(walk_dir)
# print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

collected = []

for root, subdirs, files in os.walk(walk_dir):
    # for subdir in subdirs:
    #     print('\t- subdirectory ' + subdir)

    for filename in files:
        file_path = os.path.join(root, filename)

        if not('spoken' in file_path)\
            and ('.txt' in file_path)\
            and not ('letter' in file_path)\
            and not ('slate' in file_path)\
            and not ('biomed' in file_path):
            collected.append(file_path)

print(len(collected),'collected')

def sanitize(text):
    text = re.sub(r'\r','\n',text)
    text = re.sub(r'\n{1,}','\n',text)
    text = re.sub(r'\t','',text)
    text = re.sub(r'\ {1,}',' ',text)

    text = re.sub(r'\n\ ','\n',text)
    text = re.sub(r'\ \n','\n',text)

    text = re.sub(r'\n{1,}','\n',text)
    return text

def read_one(filepath):
    with codecs.open(filepath, 'r','utf-8') as f:
        text = f.read()
        text = sanitize(text)
    return text

def show_random():
    t = random.choice(collected)
    text = read_one(t)
    print(text)
    print(t)

def into_one():
    random.shuffle(collected)
    global bigtext
    bigtext = u''
    for i in collected:
        print('reading:',i)
        bigtext += read_one(i)

    print(len(bigtext))

def save():
    with codecs.open('oanc_partial.txt','w','utf-8') as f:
        f.write(bigtext)
        
