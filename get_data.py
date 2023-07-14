import json, time, datetime, math, csv, copy, sys
import numpy as np
import pickle
from dateutil.parser import parse
from tqdm import tqdm

# read worker attribute: worker_quality
worker_quality = {}
csvfile = open("worker_quality.csv", "r")
csvreader = csv.reader(csvfile)
next(csvreader)
for line in csvreader:
    if float(line[1]) > 0.0:
        worker_quality[int(line[0])] = float(line[1]) / 100.0
    else:
        worker_quality[int(line[0])] = 0.0
csvfile.close()

# read project id
file = open("project_list.csv", "r")
project_list_lines = file.readlines()
file.close()
project_dir = "project/"
entry_dir = "entry/"

all_begin_time = parse("2018-01-01T0:0:0Z")

project_info = {}
entry_info = {}
answer_info = {}
limit = 24

average_score_list = []
client_feedback_list = []
total_awards_and_tips_list = []

project_id2index_dict = {}
project_index2id_dict = {}
worker_id2index_dict = {}
worker_index2id_dict = {}
worker_entry_history_dict = {}

industry_dict = {}
sub_category_dict = {}
category_dict = {}

worker_list = []
worker_time_list = []

for line in tqdm(project_list_lines):
    line = line.strip('\n').split(',')

    project_id = int(line[0])
    entry_count = int(line[1])

    file = open(project_dir + "project_" + str(project_id) + ".txt", "r")
    htmlcode = file.read()
    file.close()
    text = json.loads(htmlcode)

    project_info[project_id] = {}

    sub_category = int(text["sub_category"])
    category = int(text["category"])
    industry = text["industry"]
    

    if project_id not in project_id2index_dict:
        project_id2index_dict[project_id] = len(project_id2index_dict)
        project_index2id_dict[project_id2index_dict[project_id]] = project_id
    if sub_category not in sub_category_dict:
        sub_category_dict[sub_category] = len(sub_category_dict)
    if category not in category_dict:
        category_dict[category] = len(category_dict)
    if industry not in industry_dict:
        industry_dict[industry] = len(industry_dict)

    project_info[project_id]["sub_category"] = sub_category # project sub_category
    project_info[project_id]["category"] = category # project category
    project_info[project_id]["entry_count"] = int(text["entry_count"]) # project answers in total
    project_info[project_id]["start_date"] = parse(text["start_date"]) # project start date
    project_info[project_id]["deadline"] = parse(text["deadline"]) # project end date

    project_info[project_id]["average_score"] = float(text["average_score"])
    project_info[project_id]["client_feedback"] = float(text["client_feedback"])
    project_info[project_id]["total_awards_and_tips"] = float(text["total_awards_and_tips"])
    average_score_list.append(float(text["average_score"]))
    client_feedback_list.append(float(text["client_feedback"]))
    total_awards_and_tips_list.append(float(text["total_awards_and_tips"]))

    project_info[project_id]["industry"] = industry # project domain

    entry_info[project_id] = {}
    answer_info[project_id] = {}
    k = 0
    while (k < entry_count):
        file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r")
        htmlcode = file.read()
        file.close()
        text = json.loads(htmlcode)

        for item in text["results"]:
            entry_number = int(item["entry_number"])
            worker_number = int(item["author"])
            if worker_number not in worker_id2index_dict:
                worker_id2index_dict[worker_number] = len(worker_id2index_dict)
                worker_index2id_dict[worker_id2index_dict[worker_number]] = worker_number
            if worker_number not in answer_info[project_id]:
                answer_info[project_id][worker_number] = {
                    "winner": False,
                    "finalist": False,
                    "award_value": 0.0,
                    "tip_value": 0.0,
                    "answer_cnt": 0
                }
            answer_info[project_id][worker_number]["winner"] = answer_info[project_id][worker_number]["winner"] or bool(item["winner"])                                                       
            answer_info[project_id][worker_number]["finalist"] = answer_info[project_id][worker_number]["finalist"] or bool(item["finalist"])
            if item["award_value"] is not None:
                answer_info[project_id][worker_number]["award_value"] += float(item["award_value"])
            if item["tip_value"] is not None:
                answer_info[project_id][worker_number]["tip_value"] += float(item["tip_value"])
            answer_info[project_id][worker_number]["answer_cnt"] += 1

            worker_list.append(int(item["author"]))
            worker_time_list.append(parse(item["entry_created_at"]))

            if worker_number not in worker_entry_history_dict:
                worker_entry_history_dict[worker_number] = list()
            worker_entry_history_dict[worker_number].append((parse(item["entry_created_at"]), project_id))

            entry_info[project_id][entry_number] = {}
            entry_info[project_id][entry_number]["entry_created_at"] = parse(item["entry_created_at"]) # worker answer_time
            entry_info[project_id][entry_number]["worker"] = worker_number # work_id

            entry_info[project_id][entry_number]["winner"] = bool(item["winner"])
            entry_info[project_id][entry_number]["finalist"] = bool(item["finalist"])
            entry_info[project_id][entry_number]["award_value"] = float(item["award_value"]) if item["award_value"] is not None else None
            entry_info[project_id][entry_number]["tip_value"] = float(item["tip_value"]) if item["tip_value"] is not None else None

        k += limit

worker_time_list, worker_list = zip(*sorted(zip(worker_time_list, worker_list)))

for project_id, _ in project_info.items():
    project_info[project_id]["average_score"] = (
        project_info[project_id]["average_score"] - min(average_score_list)
    ) / (max(average_score_list) - min(average_score_list))
    project_info[project_id]["client_feedback"] = (
        project_info[project_id]["client_feedback"] - min(client_feedback_list)
    ) / (max(client_feedback_list) - min(client_feedback_list))
    project_info[project_id]["total_awards_and_tips"] = (
        project_info[project_id]["total_awards_and_tips"] - min(total_awards_and_tips_list)
    ) / (max(total_awards_and_tips_list) - min(total_awards_and_tips_list))

print("finish read_data")

for worker_id, history_list in worker_entry_history_dict.items():
    worker_entry_history_dict[worker_id] = sorted(history_list)

new_worker_entry_history_dict = dict()
for worker_id, history_list in worker_entry_history_dict.items():
    new_worker_entry_history_dict[worker_id] = list()
    entry_time_list, project_id_list = zip(*history_list)
    vis_project = dict()
    for entry_time, project_id in history_list:
        if project_id in vis_project:
            # answer_info[project_id][worker_id]["answer_cnt"] += 1
            continue
        vis_project[project_id] = 1
        # answer_info[project_id][worker_id]["answer_cnt"] = 1
        new_worker_entry_history_dict[worker_id].append((entry_time, project_id))
len(new_worker_entry_history_dict)

prior_worker_history_dict = dict()
train_worker_history_dict = dict()
valid_worker_history_dict = dict()
test_worker_history_dict = dict()
for worker_id, history_list in new_worker_entry_history_dict.items():
    cnt = 0
    prior_len = int(len(history_list) * 0.2)
    train_len = int(len(history_list) * 0.8)
    valid_len = int(len(history_list) * 0.9)
    prior_worker_history_dict[worker_id] = list()
    train_worker_history_dict[worker_id] = list()
    valid_worker_history_dict[worker_id] = list()
    test_worker_history_dict[worker_id] = list()
    for entry_time, project_id in history_list:
        if cnt < prior_len:
            prior_worker_history_dict[worker_id].append((entry_time, project_id))
        elif cnt < train_len:
            train_worker_history_dict[worker_id].append((entry_time, project_id))
        elif cnt < valid_len:
            valid_worker_history_dict[worker_id].append((entry_time, project_id))
        else:
            test_worker_history_dict[worker_id].append((entry_time, project_id))
        cnt += 1
print(len(prior_worker_history_dict))
print(len(train_worker_history_dict))
print(len(valid_worker_history_dict))
print(len(test_worker_history_dict))

split_data = prior_worker_history_dict, train_worker_history_dict, valid_worker_history_dict, test_worker_history_dict
with open("split_data.pickle", "wb") as f:
    pickle.dump(split_data, f, 1)

data = worker_quality, project_info, answer_info, \
    worker_time_list, worker_list, project_id2index_dict, project_index2id_dict, \
        worker_id2index_dict, worker_index2id_dict, industry_dict, sub_category_dict, category_dict
with open("env_data.pickle", "wb") as f:
    pickle.dump(data, f, 1)