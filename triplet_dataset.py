import torch
import cv2
import os
import random
import pdb
class RealMaskedFacesDataset_Triplet(torch.utils.data.Dataset):
    def __init__(self, split_file, transforms=None):
        self.path_to_real_images_yesmask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
        self.path_to_real_images_nomask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_face_dataset'
        self.samples = []
        self.dict = {0:[], 1:[]}
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        #get anchor
        path_to_anchor, label = self.samples[index]
        # pdb.set_trace()

        #get a different label
        negative_label = label
        while negative_label==label:
            negative_label = list(self.dict.keys())[random.randint(0, len(self.dict.keys()) -1)]

        #lst = [x for x in range(3)]
        #lst.remove(label), then pick

        #get different path same label as positive

        path_to_positive = self.dict[label][random.randint(0, len(self.dict[label])-1)]

        #get different label img as negative
        path_to_negative = self.dict[negative_label][random.randint(0, len(self.dict[negative_label])-1)]

        #return a tuple
        anchor, positive, negative = cv2.imread(path_to_anchor), cv2.imread(path_to_positive), cv2.imread(path_to_negative)
        anchor, positive, negative = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB), cv2.cvtColor(positive, cv2.COLOR_BGR2RGB), cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
        if self.transforms:
            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

        return anchor, positive, negative, label

    def __len__(self):
        return len(self.samples)

    def _init(self):
        name_file = open(self.split_file, 'r')
        names = []
        for name in name_file:
            names.append(name.strip())
        name_file.close()

        for path_, label in [(self.path_to_real_images_nomask, 0), (self.path_to_real_images_yesmask, 1)]:
            for name in names:
                if not os.path.exists(os.path.join(path_, name)):
                    continue
                for inst in os.listdir(os.path.join(path_, name)):
                    self.samples.append((os.path.join(path_, name, inst), label))
                    self.dict[label].append(os.path.join(path_, name, inst))

        # names = set(os.listdir(self.path_to_real_images_nomask).extend(os.listdir(self.path_to_real_images_yesmask)))
        # for name in names:
        #     self.dict[name] = []
        # for name in os.listdir(self.path_to_real_images_nomask):
        #     for img in os.path.join(self.path_to_real_images_nomask, name):
        #         self.dict[name].append(os.path.join(self.path_to_real_images_nomask, name, img))
