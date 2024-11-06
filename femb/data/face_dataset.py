import torch

from imageio import imread
import torchvision
import numpy as np
from facenet_pytorch import MTCNN

class FaceDataset(torch.utils.data.Dataset):

    def __init__(self, name=None, root='datasets/', transform=None, preprocess=None):
        assert name
        self.name = name
        self.root = root
        self.transform = transform
        self.preprocess = preprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)


        self.img_paths = None
        self.img_id_labels = None


    def __len__(self):
        return self.get_n_images()


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        img_path = self.img_paths[idx]
        img_label = self.img_id_labels[idx]

        img = imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        # [ys] add preprocessing
        if self.preprocess is not None:
            img = self.preprocess(img, mtcnn=self.mtcnn)

        return img, img_label


    def get_n_identities(self):
        return len(np.unique(self.img_id_labels))


    def get_n_images(self):
        return len(self.img_paths)


    def reduce_to_N_identities(self, N, shuffle=False):
        assert N > 0 and N <= len(self)

        id_labels = self.img_id_labels
        ids = np.unique(self.img_id_labels)

        chosen_ids = np.random.choice(ids, N, replace=False) if shuffle else ids[:N]

        chosen_idxs = np.where(np.isin(id_labels, chosen_ids))[0]

        label_map = {id: n for n, id in enumerate(chosen_ids)}

        self.img_id_labels = [label_map[self.img_id_labels[idx]] for idx in chosen_idxs]
        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]


    def reduce_to_sample_idxs(self, idxs):
        chosen_ids = np.unique(np.array(self.img_id_labels)[idxs])
        chosen_idxs = idxs
        label_map = {id: n for n, id in enumerate(chosen_ids)}

        self.img_id_labels = [label_map[self.img_id_labels[idx]] for idx in chosen_idxs]
        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]
