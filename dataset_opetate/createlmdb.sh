# 生成ssd需要的lmdb文件""

cur_dir =$(cd $( dirname ${BASH_SOURCE[0]}) & & pwd )
root_dir = / home / xn / caffe

cd $root_dir

redo = 1
data_root_dir = "$HOME/caffe/image"
dataset_name = "VEDAI"
mapfile = "$data_root_dir/VEDAI/labelmap_1024.prototxt"
anno_type = "detection"
db = "lmdb"
min_dim = 0
max_dim = 0
width = 0
height = 0

extra_cmd = "--encode-type=png --encoded"
if [ $redo]
then
extra_cmd = "$extra_cmd --redo"
fi
for subset in test_1024 trainval_1024
do
python $root_dir / scripts / create_annoset.py - -anno - type =$anno_type - -label - map - file =$mapfile - -min - dim =$min_dim - -max - dim =$max_dim - -resize - width =$width - -resize - height =$height - -check - label $extra_cmd $data_root_dir $root_dir / image /$dataset_name /$subset.txt $data_root_dir /$dataset_name /$db /$dataset_name
"_"$subset
"_"$db
examples /$dataset_name
done