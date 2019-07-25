import json


def load():
    with open('captions_train2014.json', mode='rb') as file:
        x = json.load(file)
    return x
