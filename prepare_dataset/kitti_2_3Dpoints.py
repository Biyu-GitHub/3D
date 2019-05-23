'''
转换kitti标签，用于3D点回归
转换后的格式为 xmin, ymin, max(w , h), (fblx, fbly, fbrx, fbry, rblx, rbly, ftly)/max(w , h)
'''
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


CROP_H = 352
CROP_W = 1216

fix_img_out = "../fix_img/"


def get_args():
    parser = argparse.ArgumentParser(
        description='Convert KITTI Label to Master Thesis')

    parser.add_argument("--image_dir", default="../kitti/training/image_2/",
                        help="the root path of kitti/image_2")

    parser.add_argument("--box2d_path", default="../kitti/training/label_2/",
                        help="the root path of kitti/label_2")

    parser.add_argument("--calib_dir", default="../kitti/training/calib/",
                        help="the root path of kitti/calib")

    parser.add_argument("--out_path", default="../fix_label/",
                        help="the out path of newlabel")

    parser.add_argument("--classes", default=['Car', 'Van', 'Truck'],
                        help="the root path of kitti/calib")

    return parser.parse_args()


def get_box_3d(line, cam_to_img, fix_x, fix_y):
    '''
    :param line: 一个标签文件的一行，按照空格分割好的列表
    :param cam_to_img: 相机矩阵
    :return: 存储8个点的列表box_3d
    '''
    dims = np.asarray([float(number) for number in line[8:11]])
    center = np.asarray([float(number) for number in line[11:14]])

    rot_y = float(line[3]) + np.arctan(center[0] / center[2])

    box_3d = []

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:
                point = np.copy(center)
                point[0] = center[0] + i * dims[1] / 2 * np.cos(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.cos(
                    -rot_y)
                point[2] = center[2] + i * dims[1] / 2 * np.sin(-rot_y + np.pi / 2) + (j * i) * dims[2] / 2 * np.sin(
                    -rot_y)
                point[1] = center[1] - k * dims[0]

                point = np.append(point, 1)
                point = np.dot(cam_to_img, point)
                point = point[:2] / point[2]
                # point = point.astype(np.int16)

                point[0] -= fix_x
                point[1] -= fix_y
                box_3d.append(point)

    return box_3d


def get_box_2d(line, fix_x, fix_y):
    '''
    :param line: kitti标签中的一行
    :return: center_x, center_y, w, h
    '''
    xmin = float(line[4]) - fix_x
    ymin = float(line[5]) - fix_y
    xmax = float(line[6]) - fix_x
    ymax = float(line[7]) - fix_y

    c_x = (xmin + xmax) / 2.
    c_y = (ymin + ymax) / 2.

    return [c_x, c_y, xmax - xmin, ymax - ymin]


def display():
    for f in os.listdir(args.out_path):
        box3d_file = os.path.join(args.out_path, f)
        image_name = os.path.join(fix_img_out, f.replace('txt', 'png'))

        fr = open(box3d_file, "r")
        lines = fr.readlines()
        fr.close()

        if len(lines) == 0:
            continue

        img = cv2.imread(image_name)
        print(image_name, img.shape)

        for line in lines:
            line = line.strip().split()

            line = [int(float(x)) for x in line]

            cv2.line(img, (line[4], line[5]),
                     (line[6], line[7]), (0, 0, 255), 2)
            cv2.line(img, (line[4], line[5]),
                     (line[8], line[9]), (0, 0, 255), 2)
            cv2.line(img, (line[4], line[5]),
                     (line[4], line[10]), (0, 0, 255), 2)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(img[..., ::-1])
        # plt.show()
        cv2.imshow("3D", img)
        cv2.waitKey(500)


def main():
    for f in os.listdir(args.box2d_path):
        box2d_file = os.path.join(args.box2d_path, f)
        calib_file = os.path.join(args.calib_dir, f)

        image_name = os.path.join(args.image_dir, f.replace('txt', 'png'))
        fix_image_name = os.path.join(fix_img_out, f.replace('txt', 'png'))

        out_name = os.path.join(args.out_path, f)

        print(image_name)
        fw = open(out_name, "w")

        # =====================修复图片尺寸===================== #
        img = cv2.imread(image_name)
        h, w, c = img.shape

        fix_x = int((w - CROP_W) / 2)
        fix_y = h - CROP_H

        fix_img = img[fix_y:, fix_x: fix_x + CROP_W, :]
        cv2.imwrite(fix_image_name, fix_img)
        print(fix_img.shape)
        # ====================================================== #

        for line in open(box2d_file):
            line = line.strip().split(" ")

            if not line[0] in args.classes:
                continue

            for calib_line in open(calib_file):
                if 'P2:' in calib_line:
                    cam_to_img = calib_line.strip().split(' ')
                    cam_to_img = np.asarray([float(number)
                                             for number in cam_to_img[1:]])
                    cam_to_img = np.reshape(cam_to_img, (3, 4))

            box_3d = get_box_3d(line, cam_to_img, fix_x, fix_y)
            box_2d = get_box_2d(line, fix_x, fix_y)

            write_box_2d = str(
                box_2d[0]) + " " + str(box_2d[1]) + " " + str(box_2d[2]) + " " + str(box_2d[3])

            write_box_3d = str(box_3d[0][0]) + " " + str(box_3d[0][1]) + " " + str(box_3d[6][0]) + " " + str(
                box_3d[6][1]) + " " + str(box_3d[2][0]) + " " + str(box_3d[2][1]) + " " + str(box_3d[1][1])

            write_line = write_box_2d + " " + write_box_3d + "\n"

            fw.write(write_line)

        fw.close()


if __name__ == "__main__":

    args = get_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # main()
    display()
