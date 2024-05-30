# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import pathlib
import random
import sys
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes
from src.utils.labelmap import label_map
from src.utils.posemap import kpoint_to_heatmap


class DressCodeDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 multimodal_data_path: str,
                 num_test_image: int,
                 phase: str,
                 tokenizer,
                 radius=5,
                 caption_folder='fine_captions.json',
                 coarse_caption_folder='coarse_captions.json',
                 sketch_threshold_range: Tuple[int, int] = (20, 127),
                 order: str = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'im_sketch', 'captions',
                                           'original_captions', 'category', 'stitch_label'),
                 category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
                 size: Tuple[int, int] = (512, 384),
                 ):

        super(DressCodeDataset, self).__init__()
        self.dataroot = pathlib.Path(dataroot_path)
        self.phase = phase
        self.multimodal_data_path = pathlib.Path(multimodal_data_path)
        self.num_test_image = num_test_image
        self.caption_folder = caption_folder
        self.sketch_threshold_range = sketch_threshold_range
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []
        multimodal_data_path_names = []

        possible_outputs = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton',
                            'im_mask', 'inpaint_mask', 'parse_mask_total', 'cloth_sketch', 'im_sketch', 'captions',
                            'original_captions', 'category', 'hands', 'parse_head_2', 'stitch_label']

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(self.multimodal_data_path / self.caption_folder) as f:
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        with open(self.multimodal_data_path / coarse_caption_folder) as f:
            self.captions_dict.update(json.load(f))

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = self.dataroot / c
            multimodal_data_path = self.multimodal_data_path / c

            if phase == 'train':
                filename = dataroot / f"{phase}_pairs.txt"
            else:
                filename = dataroot / f"{phase}_pairs_{order}.txt"

            with open(filename, 'r') as f:
                if num_test_image > 0: #limit number of image gen by num_test_image
                    i = 0

                    for line in f.readlines():
                        if i > num_test_image:
                            break

                        im_name, c_name = line.strip().split()
                        if c_name.split('_')[0] not in self.captions_dict:
                            continue

                        im_names.append(im_name)
                        c_names.append(c_name)
                        dataroot_names.append(dataroot)

                        i += 1
                else: #run full test data, gen full image
                    for line in f.readlines():
                        im_name, c_name = line.strip().split()
                        if c_name.split('_')[0] not in self.captions_dict:
                            continue

                        im_names.append(im_name)
                        c_names.append(c_name)
                        dataroot_names.append(dataroot)
                        multimodal_data_path_names.append(multimodal_data_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.multimodal_data_path_names = multimodal_data_path_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """

        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        multimodal_data_path = self.multimodal_data_path_names[index]

        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist or "original_captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            # if train randomly shuffle captions if there are multiple, else concatenate with comma
            if self.phase == 'train':
                random.shuffle(captions)
            captions = ", ".join(captions)

            original_captions = captions

        if "captions" in self.outputlist:
            cond_input = self.tokenizer([captions], max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
            cond_input = cond_input.squeeze(0)
            max_length = cond_input.shape[-1]
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            ).input_ids.squeeze(0)
            captions = cond_input

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            image = Image.open(dataroot / 'images' / im_name)

            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "im_sketch" in self.outputlist:

            if "unpaired" == self.order and self.phase == 'test':  # Upper of multigarment is the same of unpaired
                im_sketch = Image.open(
                    multimodal_data_path / 'im_sketch_unpaired' / f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}')
            else:
                im_sketch = Image.open(multimodal_data_path / 'im_sketch' / c_name.replace(".jpg", ".png"))

            im_sketch = im_sketch.resize((self.width, self.height))
            im_sketch = ImageOps.invert(im_sketch)
            # threshold grayscale pil image
            im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
            # im_sketch = im_sketch.convert("RGB")
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
            im_sketch = 1 - im_sketch

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist or "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist or "pose_map" in self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist:
            # Label Map
            parse_name = im_name.replace('_0.jpg', '_4.png')
            im_parse = Image.open(dataroot / 'label_maps' / parse_name)
            im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(np.float32)

            parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            category = str(dataroot.name)
            if category == 'dresses':
                label_cat = 7
                parse_cloth = (parse_array == 7).astype(np.float32)
                parse_mask = (parse_array == 7).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            elif category == 'upper_body':
                label_cat = 4
                parse_cloth = (parse_array == 4).astype(np.float32)
                parse_mask = (parse_array == 4).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                     (parse_array == label_map["pants"]).astype(np.float32)

                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            elif category == 'lower_body':
                label_cat = 6
                parse_cloth = (parse_array == 6).astype(np.float32)
                parse_mask = (parse_array == 6).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                     (parse_array == 14).astype(np.float32) + \
                                     (parse_array == 15).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            else:
                raise NotImplementedError

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            # dilation
            parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
            shape = self.transform2D(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('_0.jpg', '_2.json')
            with open(dataroot / 'keypoints' / pose_name, 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []
            for pose_d in pose_data:
                ux = pose_d[0] / 384.0
                uy = pose_d[1] / 512.0

                # scale posemap points
                px = ux * self.width
                py = uy * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            if category == 'dresses' or category == 'upper_body' or category == 'lower_body':
                with open(dataroot / 'keypoints' / pose_name, 'r') as f:
                    data = json.load(f)
                    shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                    shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                    elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                    elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                    wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                    wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                if category == 'dresses' or category == 'upper_body':
                    parse_mask += im_arms
                    parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            if category == 'dresses' or category == 'upper_body':
                with open(dataroot / 'keypoints' / pose_name, 'r') as f:
                    data = json.load(f)
                    points = []
                    points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                    points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                    x_coords, y_coords = zip(*points)
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    for i in range(parse_array.shape[1]):
                        y = i * m + c
                        parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            # tune the amount of dilation here
            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            # here we have to modify the mask and get the bounding box
            bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
            bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
            xmin = bboxes[0, 0]
            xmax = bboxes[0, 2]
            ymin = bboxes[0, 1]
            ymax = bboxes[0, 3]

            inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
                torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
                torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

            inpaint_mask = inpaint_mask.unsqueeze(0)
            im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
            parse_mask_total = parse_mask_total.numpy()
            parse_mask_total = parse_array * parse_mask_total
            parse_mask_total = torch.from_numpy(parse_mask_total)

        if "stitch_label" in self.outputlist:
            stitch_labelmap = Image.open(self.multimodal_data_path / 'test_stitchmap' / im_name.replace(".jpg", ".png"))
            stitch_labelmap = transforms.ToTensor()(stitch_labelmap) * 255
            stitch_label = stitch_labelmap == 13

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        # Output interpretation
        # "c_name" -> filename of inshop cloth
        # "im_name" -> filename of model with cloth
        # "cloth" -> img of inshop cloth
        # "image" -> img of the model with that cloth
        # "im_cloth" -> cut cloth from the model
        # "im_mask" -> black mask of the cloth in the model img
        # "cloth_sketch" -> sketch of the inshop cloth
        # "im_sketch" -> sketch of "im_cloth"
        # ...
        return result

    def __len__(self):
        return len(self.c_names)
