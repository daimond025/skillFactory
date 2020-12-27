import json
import collections

with open('data.json', 'rb') as infile:
    data = json.load(infile)
data_list = data['events_data']

actions = []
for item in data_list:
    if item['category'] == 'report':
        actions.append(item['client_id'])

c = collections.Counter()
for action in actions:
    c[action] += 1

print(sorted(c.keys()))
