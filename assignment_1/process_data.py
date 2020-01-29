import sys, csv

if len(sys.argv) < 4:
  print('Expected 3 arguments got {}'.format(len(sys.argv)))
  print('Usage: python process_data.py <training set %> <test set%> <output file name>')
  quit()

training = float(sys.argv[1])
test = float(sys.argv[2])
file_name = sys.argv[3]

with open('processed.cleveland.data', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)

  for row in data:
    if '?' in row:
      data.remove(row)

  