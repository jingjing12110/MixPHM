# @File :tools.py
# @Time :2022/6/22
# @Desc :
import re
import json
import os


# *********************************************************************
def read_txt(file):
    with open(file, "r") as f:
        data = [line.strip('\n') for line in f.readlines()]
    return data


def write_txt(file, s):
    with open(file, 'a+') as f:
        f.write(s)


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


