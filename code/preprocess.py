import pandas as pd
from os import listdir
import re

path = "data/sotu/sotu_files/"

data_list = open("data/sotu/stateoftheunion1790-2017.txt",'r').read().split('***')

for i in range(1,len(data_list)-1):
	with open(path+str(i)+".txt",'w') as f:
		f.write(data_list[i])

all_files = [f for f in listdir(path)]

for i in all_files:
    with open(path+i, "r+") as f:
        data = f.read()
        data = re.sub("[^a-zA-Z\. ]+", " ", data)
        f.seek(0)
        f.write(data)
        f.truncate()

