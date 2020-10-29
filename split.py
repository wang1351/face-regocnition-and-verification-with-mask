import os
import numpy as np

# sim_lst = []
# for i in range(10):
#     sim_lst.append([])
processed_names = []

pretrain = []
pretrain_val = []

path = '/Users/wzy/Desktop/mask/maskrecognition/lfw'
for name in os.listdir(path):
    if name in processed_names:
        continue
    a = np.random.rand()
    if a < 0.7:
        pretrain.append(name)
    else:
        pretrain_val.append(name)
    processed_names.append(name)


writer = open('pretrain.txt', 'w')
for name in pretrain:
    writer.write(name + '\n')
writer.close()

writer = open('pretrain_val.txt', 'w')
for name in pretrain_val:
    writer.write(name + '\n')
writer.close()



