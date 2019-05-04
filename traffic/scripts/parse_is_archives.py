# Copyright (c) Aleksandr Fedorov

import requests
import time
import wget
import cv2
import os
import json
import copy
import subprocess
import numpy as np
from datetime import datetime
from tqdm import tqdm

# IS 74 requisites
API_URL = 'https://stream1.is74.ru/api/'
LOGIN = ""
PASSWORD = ""

DATE = "2019-04-24"
TARGET_FOLDER = f"/home/fedorov/detection/data/test-statistics/{DATE}"
IMAGE_MASK = "traffic/mask.png"
TARGET_WIDTH, TARGET_HEIGHT = 1280, 512

headers = {'Content-type': 'application/x-www-form-urlencoded',}
login_data = {
    'auth': True,
    'l': LOGIN,
    'p': PASSWORD,
    'r': False
}
search_params = {
    'r_search': True,
    'cam': "pob-mol",
    's_time': None,
    'e_time': None
}

s = requests.Session()
p = s.post(API_URL, data=login_data, headers=headers)
records = []
for hour in tqdm(range(23)):
    search_params['s_time'] = f"{DATE} {hour:0=2}:00"
    search_params['e_time'] = f"{DATE} {hour + 1:0=2}:55"
    r = s.post(API_URL, data=search_params, headers=headers)
    records.append(r.json())
    time.sleep(2)

# API is a bit buggy, last hours must be fetched explicitly
search_params['s_time'] = f"{DATE} 22:45"
search_params['e_time'] = f"{DATE} 23:59"
r = s.post(API_URL, data=search_params, headers=headers)
records.append(r.json())


# Merge all fetched data
tmp = {}
for d in records:
    tmp.update(d)
records = tmp
print(f"Fetched {len(records)} records")

# Date patter used by server
pattern = "%Y-%m-%d %H:%M:%S"
for k, v in records.items():
    v['stime'] = datetime.strptime(v['stime'], pattern)
    v['etime'] = datetime.strptime(v['etime'], pattern)
    v['url'] = f"https://stream1.is74.ru/rec/{k}?dl=1"  # URL to download


# Split data into continious groups
records = sorted(list(records.values()), key=lambda x:x['stime'])
groups = [[]]
for r in records:
    if len(groups[-1]) == 0:
        groups[-1].append(r)
        continue
    if r['stime'] == groups[-1][-1]['etime']:
        groups[-1].append(r)
    else:
        groups.append([r])
print(f"Records form {len(groups)} continious groups")


CREATED_FILES = []
for group in groups:
    begin, end = group[0]['stime'], group[-1]['etime']
    begin, end = begin.time().isoformat(), end.time().isoformat()
    print(f"Processing data from {begin} to {end}")

    folder = os.path.join(TARGET_FOLDER, f"{begin}-{end}")

    out = os.path.join(folder, f"video.mp4")
    CREATED_FILES.append(out)

    if os.path.exists(out):
        print(f"Skipping {folder}, exists and parsed")
        continue

    os.makedirs(folder, exist_ok=True)
    print(f"Created folder {folder}")

    concat_order = []
    for ind, f in tqdm(enumerate(group)):
        name = os.path.join(folder, f"{ind}.mp4")
        if os.path.exists(name):
            continue

        # Repeat attentions to download till success
        downloaded = False
        while not downloaded:
            try:
                wget.download(f['url'], out=name)
                downloaded = True
            except Exception as e:
                print("Error :", e)
                time.sleep(10)

        concat_order.append(name)

    tmp_file = os.path.join(folder, 'order.txt')
    with open(tmp_file, 'w') as file:
        for o in concat_order:
            print(f"file {o}", file=file)

    ret = subprocess.call(f'ffmpeg -f concat -safe 0 -i "{tmp_file}" -c copy "{out}"', shell=True)
    assert ret == 0

    # Clean up
    for o in concat_order:
        os.remove(o)
    os.remove(tmp_file)

    meta_file = os.path.join(folder, "meta.json")
    meta = {
        "stime": group[0]['stime'].isoformat(),
        "etime": group[-1]['etime'].isoformat(),
        "files": copy.deepcopy(group)
    }
    for v in meta["files"]:
        v['stime'] = v['stime'].isoformat()
        v['etime'] = v['etime'].isoformat()
    with open(meta_file, "w") as file:
        json.dump(meta, file, indent=4, sort_keys=False)


IMAGE_MASK = cv2.imread(IMAGE_MASK, cv2.IMREAD_UNCHANGED)
data_points = np.argwhere(IMAGE_MASK)
min_y, min_x = data_points.min(axis=0)
max_y, max_x = data_points.max(axis=0) + 1
IMAGE_MASK = IMAGE_MASK[min_y:max_y, min_x:max_x]
assert IMAGE_MASK.shape == (769, 1920)

x,y = min_x, min_y
wcrop, hcrop = max_x-min_x, max_y-min_y


# Resize files
for file in CREATED_FILES:
    print(f"Rescaling file {file}")
    folder = os.path.dirname(file)
    resized = os.path.join(folder, f"video_{TARGET_WIDTH}x{TARGET_HEIGHT}.mp4")
    if os.path.exists(resized):
        continue
    ret = subprocess.call(f'ffmpeg -i "{file}" -vf '
                          f'"crop={wcrop}:{hcrop}:{min_x}:{min_y}, scale={TARGET_WIDTH}:{TARGET_HEIGHT}" '
                          f'"{resized}"', shell=True)
    assert ret == 0
