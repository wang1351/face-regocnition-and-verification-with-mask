import os
import numpy as np

train_names = []
val_names = []
test_names = []
processed_names = []

path = ''
for path_ in [path]:
   for name in os.listdir(path_):
       if name in processed_names:
         continue
       a = np.random.rand()
       if a < 0.7:
           train_names.append(name)
       elif a < 0.8:
           val_names.append(name)
       else:
            test_names.append(name)
       processed_names.append(name)


train_writer = open('train.txt', 'w')
for name in train_names:
  train_writer.write(name+'\n')
train_writer.close()
val_writer = open('val.txt', 'w')
for name in val_names:
  val_writer.write(name+'\n')
val_writer.close()

test_writer = open('test.txt', 'w')
for name in test_names:
  test_writer.write(name+'\n')
test_writer.close()
