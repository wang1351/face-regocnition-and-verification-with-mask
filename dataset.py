import torch
import cv2
import os
import pdb


# dataset: each epoch only part of simulated!
class SimulationMaskedFacesDataset_Classification(torch.utils.data.Dataset):
    def __init__(self,split_file, partition, transforms=None):
        self.path_to_real_images_yesmask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
        self.path_to_real_images_nomask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_face_dataset'
        self.samples = []
        self.path = '/Users/wzy/Desktop/mask/maskrecognition/CASIA-WebFace_masked/webface_masked'
        self.split_file = split_file
        self.transforms = transforms
        self.partition = partition
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _init(self):
        name_file = open(self.split_file,'r')
        names = []
        for name in name_file:
            names.append(name.strip())
        name_file.close()

        number_file = open(str('sim'+str(self.partition)+'.txt'), 'r')
        numbers = []
        for number in number_file:
            numbers.append(number.strip())
        number_file.close()

        for path_, label in [(self.path_to_real_images_nomask,0), (self.path_to_real_images_yesmask,1)]:
            for name in names:
                if not os.path.exists(os.path.join(path_, name)):
                    continue
                for inst in os.listdir(os.path.join(path_, name)):
                    self.samples.append((os.path.join(path_,name,inst),label))

        for number in numbers:
            if not os.path.exists(os.path.join(self.path, number)):
                continue
            for inst in os.listdir(os.path.join(self.path, number)):
                self.samples.append((os.path.join(self.path, number, inst), 1))



class SimulationMaskedFacesDataset_Classification_withsplit(torch.utils.data.Dataset):
    def __init__(self, split_file):
        self.path = '/Users/wzy/Desktop/mask/maskrecognition/CASIA-WebFace_masked/webface_masked'
        self.samples = []
        self.split_file = split_file
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
        img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        img = torch.from_numpy(img).permute(2,0,1)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _init(self):
        number_file = open(self.split_file, 'r')
        numbers = []
        for number in number_file:
            numbers.append(number.strip())
        number_file.close()
        for number in number_file:
            for inst in os.listdir(os.path.join(self.path, number)):
                self.samples.append((os.path.join(self.path,number,inst), 1))

class RealMaskedFacesDataset_Classification(torch.utils.data.Dataset):
    def __init__(self,split_file, transforms=None):
        self.path_to_real_images_yesmask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
        self.path_to_real_images_nomask = '/Users/wzy/Desktop/mask/maskrecognition/self-built-masked-face-recognition-dataset/AFDB_face_dataset'
        self.samples = []
        self.split_file = split_file
        self.transforms = transforms
        self._init()

    def __getitem__(self, index):
        # pdb.set_trace()
        path_to_img, label = self.samples[index]
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _init(self):
        name_file = open(self.split_file,'r')
        names = []
        for name in name_file:
            names.append(name.strip())
        name_file.close()
        for path_, label in [(self.path_to_real_images_nomask,0), (self.path_to_real_images_yesmask,1)]:
            for name in names:
                if not os.path.exists(os.path.join(path_, name)):
                    continue
                for inst in os.listdir(os.path.join(path_, name)):
                    self.samples.append((os.path.join(path_,name,inst),label))

        i = 0
        j = 0
        for x in range(len(self.samples)):
            if self.samples[i][1]==0:
                i += 1
            else:
                j += 1


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, partition, transforms=None):
        self.f_path = '/Users/wzy/Desktop/mask/maskrecognition/lfw'
        self.samples = []
        self.path = '/Users/wzy/Desktop/mask/maskrecognition/CASIA-WebFace_masked/webface_masked'
        self.transforms = transforms
        self.partition = partition
        self.split_file = split_file
        self._init()

    def __getitem__(self, index):
        path_to_img, label = self.samples[index]
        img = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def _init(self):
        number_file = open(str('sim'+str(self.partition)+'.txt'), 'r')
        numbers = []
        for number in number_file:
            numbers.append(number.strip())
        number_file.close()

        for number in numbers:
            if not os.path.exists(os.path.join(self.path, number)):
                continue
            for inst in os.listdir(os.path.join(self.path, number)):
                self.samples.append((os.path.join(self.path, number, inst), 1))

        name_file = open(self.split_file, 'r')
        names = []
        for name in name_file:
            names.append(name.strip())
        name_file.close()

        for name in names:
            for inst in os.listdir(os.path.join(self.f_path, name)):
                self.samples.append((os.path.join(self.f_path, name, inst), 0))
        print(len(names), len(numbers))