import json
from pprint import pprint

def loadJson(fileLoc):
    data_file = open(fileLoc)
    return json.load(data_file)
    #with open('model_idflickr8k.json') as data_file:    
    #    data = json.load(data_file)

