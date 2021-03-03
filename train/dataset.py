import torch
import os
import numpy as np
from PIL import Image


class CityscapesDataset(object):
    def __init__(self, root, transforms, instance=False):
        self.root = root
        self.transforms = transforms
        self.instance = instance
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # print(img_path + '  open')
        img = Image.open(img_path).convert("RGB").resize((512, 256), 1)
        # img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).resize((512, 256), 0)
        # mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        delete_idx = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if (ymax - ymin > 1 and xmax - xmin > 1):
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                delete_idx.append(i)
        if len(delete_idx) != 0:
            obj_ids = np.delete(obj_ids, delete_idx)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if self.instance:
            obj_ids = np.where(obj_ids // 1000 >= 1, obj_ids // 1000, obj_ids)
            masks = np.where(masks // 1000 >= 1, masks // 1000, masks)

        labels = torch.tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        if (boxes.size()[0] == 0):
            print(img_path, "  ", boxes.size()[0])

        if boxes.size()[0] != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # if(np.where(np.array(labels)>4)[0].any()):
        # print(mask_path , '\n' , boxes , '\n' , labels , '\n' , len(boxes), ' ', len(labels))
        return img, target

    def __len__(self):
        return len(self.imgs)