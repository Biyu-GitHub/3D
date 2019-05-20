## 转换kitti数据集标签

转换label，用于3D车辆检测

标签的输出格式为：`xmin, ymin, max(w , h), (fblx, fbly, fbrx, fbry, rblx, rbly, ftly)/max(w , h)`

```bash
IMAGE_DIR="../kitti/training/image_2/"
BOX2D_PATH="../kitti/training/label_2/"
CALIB_DIR="../kitti/training/calib/"
OUT_PATH="../newlabel1/"
CLASSES=['Car', 'Van', 'Truck']
DEBUG=False

python kitti_2_3Dpoints.py \
	--image_dir=${IMAGE_DIR} \
	--box2d_path=${BOX2D_PATH} \
	--calib_dir=${CALIB_DIR} \
	--out_path=${OUT_PATH} \
	--classes=${CLASSES} \
	--debug=${DEBUG}
```

