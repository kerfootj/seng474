import sys, csv, random, pickle

def format_data(unformated):
  data, target = [], []
  for row in unformated:
    data.append(row[:-1])
    target.append(min(int(row[-1]), 1))

  return data, target
  

if len(sys.argv) < 3:
  print('Expected 3 arguments got {}'.format(len(sys.argv)))
  print('Usage: python process_data.py <training set %> <input data file> <input name file> <output file name>')
  exit()

tr_percent = float(sys.argv[1])
in_name = sys.argv[2]
name_file = sys.argv[3]
out_name = sys.argv[4]

data = None
with open(in_name, 'r') as f:
  reader = csv.reader(f)
  temp = list(reader)

  for row in temp:
    if '?' in row:
      temp.remove(row)

  data = [[float(x) for x in row] for row in temp]

test = []
for i in range(int((1 - tr_percent) * len(data))):
  test.append(data.pop(random.randint(0, len(data) - 1)))

training_data, training_target = format_data(data)
test_data, test_target = format_data(test)

names = None
with open(name_file, 'r') as f:
  reader = csv.reader(f)
  names = list(reader)

d = {
  "training_data": training_data,
  "training_target": training_target,
  "test_data": test_data,
  "test_target": test_target,
  "feature_names": names[0],
  "class_names": names[1]
}

with open('data/' + out_name, 'wb') as f:
  pickle.dump(d, f)
