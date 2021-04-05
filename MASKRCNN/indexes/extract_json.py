file1 = open('DAYtrain.txt', 'r')
Lines = file1.readlines()
import json
import os


filenames = []

for line in Lines:
    x = line.split('\n')
    name = x[0] + '.jpg'
    filenames.append(name)

print(filenames)

file = 'via_region_data.json'


daydict = dict()

with open(file, "r") as read_file:
    data = json.load(read_file)
    keylist = data.keys()
    for key in keylist:
        stringkey = str(key)
        print(stringkey)
        if stringkey not in filenames:
            print('finding one')
            daydict[key] = data[key]

with open('345.json', 'w') as f:
    json.dump(daydict,f)