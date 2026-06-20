# encoding=utf-8

import os
import copy
import numpy as np
import cv2
# import shutil
from collections import defaultdict
from tqdm import tqdm

# We need 10 classes to detect and tracking
cls2id = {
    'car': 0,
    # 'truck': 1,
    # 'bus': 2,
}

id2cls = {
    0: 'car',
    # 1: 'truck',
    # 2: 'bus',
}


def draw_ignore_regions(img, boxes):
    """
    输入图片ignore regions涂黑
    :param img: opencv(numpy array): H×W×C
    :param boxes: a list of boxes: left(box[0]), top(box[1]), width(box[2]), height(box[3])
    :return:
    """
    if img is None:
        print('[Err]: Input image is none!')
        return -1

    for box in boxes:
        box = list(map(lambda x: int(x + 0.5), box))
        img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = [0, 0, 0]

    return img


def gen_dot_train_file(data_root, rel_path, out_root, f_name='detrac.train'):
    """
    To generate the dot train file
    :param data_root:
    :param rel_path:
    :param out_root:
    :param f_name:
    :return:
    """
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = out_root + '/' + f_name
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + rel_path
        seqs = [x for x in os.listdir(root)]
        seqs.sort()
        # seqs = sorted(seqs, key=lambda x: int(x.split('_')[-1]))
        for seq in tqdm(seqs):
            img_dir = root + '/' + seq  # + '/img1'
            rel_dir = rel_path + '/' + seq
            img_list = [x for x in os.listdir(img_dir)]
            img_list.sort()
            for img in img_list:
                if img.endswith('.jpg'):
                    img_path = img_dir + '/' + img
                    img_path_out = rel_dir + '/' + img
                    if os.path.isfile(img_path):
                        item = img_path_out.replace(data_root + '/', '')
                        # print(item)
                        f.write(item + '\n')
                        cnt += 1

    print('Total {:d} images for training'.format(cnt))


def gen_track_dataset(src_root, dst_root, viz_root=None):
    """
    :param src_root:
    :param dst_root:
    :param viz_root:
    :return:
    """
    if not os.path.isdir(src_root):
        print('[Err]: invalid sr dir.')
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    dst_img_root = dst_root + '/images'
    dst_txt_root = dst_root + '/labels_with_ids'
    if not os.path.isdir(dst_img_root):
        os.makedirs(dst_img_root)
    if not os.path.isdir(dst_txt_root):
        os.makedirs(dst_txt_root)

    # track_start_id = 0
    track_start_id_dict = defaultdict(int)
    for cls_id in id2cls.keys():
        track_start_id_dict[cls_id] = 0

    frame_cnt = 0

    seq_names = [x for x in os.listdir(src_root + '/sequences')]
    seq_names.sort()

    for seq in tqdm(seq_names):
        print('Processing {}:'.format(seq))

        seq_img_dir = src_root + '/sequences/' + seq
        seq_txt_f_path = src_root + '/annotations/' + seq + '_gt.txt'
        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_txt_f_path)):
            print('[Warning]: invalid src img dir or invalid annotations file(txt).')
            continue

        dst_seq_img_dir = dst_img_root + '/' + seq
        if not os.path.isdir(dst_seq_img_dir):
            os.makedirs(dst_seq_img_dir)
        dst_seq_txt_dir = dst_txt_root + '/' + seq
        if not os.path.isdir(dst_seq_txt_dir):
            os.makedirs(dst_seq_txt_dir)

        # seq_max_tar_id = 0
        seq_max_tra_id_dict = defaultdict(int)
        for k in id2cls.keys():
            seq_max_tra_id_dict[k] = 0

        seq_frame_names = os.listdir(seq_img_dir)
        seq_frame_names.sort()

        with open(seq_txt_f_path, 'r', encoding='utf-8') as f_r:
            label_lines = f_r.readlines()
            label_n_lines = len(label_lines)
            seq_label_array = np.zeros((label_n_lines, 9), np.int32)

            for line_i, line in enumerate(label_lines):
                line = [int(x) for x in line.strip().split(',')]
                seq_label_array[line_i] = line

        seq_ignore_box_label = seq_label_array[seq_label_array[:, 7] == 0]

        seq_obj_boxes = seq_label_array[(seq_label_array[:, 7] > 0) & (seq_label_array[:, 7] < 4)]

        seq_ignore_box_dict = defaultdict(list)
        seq_objs_label_dict = defaultdict(list)

        for label in seq_ignore_box_label:  # key: frame_id(start from 1)
            seq_ignore_box_dict[label[0]].append(label[2:6])

        for label in seq_obj_boxes:  # key: frame_id(start from 1)
            seq_objs_label_dict[label[0]].append(label)

        seq_cls_target_ids_dict = defaultdict(list)
        tmp_ids_dict = defaultdict(set)
        for fr_id in seq_objs_label_dict.keys():
            fr_labels = seq_objs_label_dict[fr_id]

            for label in fr_labels:
                # cls_id = label[7] - 1
                cls_id = 0
                target_id = label[1]

                # seq_cls_target_ids_dict[cls_id].append(target_id)  # key: cls_id
                tmp_ids_dict[cls_id].add(target_id)

        for cls_id in tmp_ids_dict.keys():
            track_ids = tmp_ids_dict[cls_id]
            # track_ids = set(track_ids)
            track_ids = list(track_ids)
            track_ids.sort()
            seq_cls_target_ids_dict[cls_id] = track_ids

        for k, v in seq_cls_target_ids_dict.items():
            seq_max_tra_id_dict[k] = len(v)
        # print('Seq {}:'.format(seq))
        for k in id2cls.keys():
            print("{} max track id: {:d}, start id: {:d}"
                  .format(id2cls[k], seq_max_tra_id_dict[k], track_start_id_dict[k]))

        for fr_id in seq_objs_label_dict.keys():
            # -----
            fr_labels = seq_objs_label_dict[fr_id]

            fr_name = 'img{:06d}.jpg'.format(fr_id)
            fr_path = seq_img_dir + '/' + fr_name
            if not os.path.isfile(fr_path):
                print('[Err]: invalid image file {}.'.format(fr_path))
                continue

            # H×W×C: BGR
            img = cv2.imread(fr_path, cv2.IMREAD_COLOR)
            if img is None:
                print('[Err]: empty image.')
                continue

            H, W, C = img.shape

            draw_ignore_regions(img, seq_ignore_box_dict[fr_id])

            # dst_img_path = dst_seq_img_dir + '/' + fr_name
            dst_img_path = dst_seq_img_dir + '/' + '{:07d}.jpg'.format(fr_id)
            if not os.path.isfile(dst_img_path):
                cv2.imwrite(dst_img_path, img)
                # print('{} saved to {}'.format(fr_path, dst_seq_img_dir))

            if not (viz_root is None):
                viz_dir = viz_root + '/' + seq
                if not os.path.isdir(viz_dir):
                    os.makedirs(viz_dir)
                viz_path = viz_dir + '/' + fr_name

                img_viz = copy.deepcopy(img)


            fr_label_strs = []
            for label in fr_labels:
                # cls_id and cls_name
                obj_type = label[7]
                assert 0 < obj_type < 11
                cls_id = obj_type - 1  # 从0开始
                # cls_name = id2cls[cls_id]

                target_id = label[1]

                track_id = seq_cls_target_ids_dict[cls_id].index(target_id) + 1 + track_start_id_dict[cls_id]
                # track_id = target_id

                bbox_left = label[2]
                bbox_top = label[3]
                bbox_width = label[4]
                bbox_height = label[5]

                score = label[6]
                truncation = label[
                    8]  # no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% °´ 50%))
                occlusion = label[-1]

                if occlusion > 1:  # heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
                    # print('[Warning]: skip the bbox because of heavy occlusion')
                    continue

                if not (viz_root is None):
                    pt_1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                    pt_2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                    cv2.rectangle(img_viz,
                                  pt_1,
                                  pt_2,
                                  (0, 255, 0),
                                  2)

                    cls_str = id2cls[cls_id]
                    veh_type_str_size = cv2.getTextSize(cls_str,
                                                        cv2.FONT_HERSHEY_PLAIN,
                                                        1.3,
                                                        1)[0]
                    cv2.putText(img_viz,
                                cls_str,
                                (pt_1[0],
                                 pt_1[1] + veh_type_str_size[1] + 8),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.3,
                                [225, 255, 255],
                                1)

                    tr_id_str = str(track_id)
                    tr_id_str_size = cv2.getTextSize(tr_id_str,
                                                     cv2.FONT_HERSHEY_PLAIN,
                                                     1.3,
                                                     1)[0]
                    cv2.putText(img_viz,
                                tr_id_str,
                                (pt_1[0],
                                 pt_1[1] + veh_type_str_size[1] + tr_id_str_size[1] + 8),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.3,
                                [225, 255, 255],
                                1)

                bbox_center_x = bbox_left + bbox_width * 0.5
                bbox_center_y = bbox_top + bbox_height * 0.5

                bbox_center_x /= W
                bbox_center_y /= H
                bbox_width /= W
                bbox_height /= H

                # class_id, track_id, bbox_center_x, box_center_y, bbox_width, bbox_height
                label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    cls_id,
                    track_id,
                    bbox_center_x,
                    bbox_center_y,
                    bbox_width,
                    bbox_height)
                fr_label_strs.append(label_str)

            if not (viz_root is None):
                cv2.imwrite(viz_path, img_viz)

            # label_f_path = dst_seq_txt_dir + '/' + fr_name.replace('.jpg', '.txt')
            label_f_path = dst_seq_txt_dir + '/' + '{:07d}.jpg'.format(fr_id).replace('.jpg', '.txt')
            with open(label_f_path, 'w', encoding='utf-8') as f:
                for label_str in fr_label_strs:
                    f.write(label_str)
            # print('{} written.'.format(label_f_path))

            frame_cnt += 1

        for cls_id in id2cls.keys():
            track_start_id_dict[cls_id] += seq_max_tra_id_dict[cls_id]
        print('Processing seq {} done.\n'.format(seq))

    print('Total {:d} frames'.format(frame_cnt))


