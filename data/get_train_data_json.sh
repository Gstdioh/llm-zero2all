#!/bin/bash

# 该脚本用于生成训练数据集
# 从多个文件夹中随机选择一定数量的json文件，并将这些文件复制到一个新的文件夹中
#   从   ./01_pre_data/baike2018qa/json_baike2018qa/中随机选择8个json文件
# 复制到 ./02_train_data/baike2018qa/中

train_data_dir="./02_train_data"

# 删除文件夹
if [ -d "./$train_data_dir" ]; then
    rm -r ./$train_data_dir
fi

# 创建新的文件夹
mkdir -p ./$train_data_dir

# 原始文件夹列表
raw_data_dir="./01_pre_data"
folder_names=("baike2018qa" "github-python" "news2016zh" "webtext2019zh" "wikipedia_cn_20240416" "wikipedia_en_20220301")

# 定义每个文件夹需要选择的json文件数量
num_files=(8 11 20 21 14 40)

# 遍历每个文件夹
for ((i=0; i<${#folder_names[@]}; i++))
do
    folder_name=${folder_names[i]}
    # # 获取文件夹名称
    # folder_name=$(basename "$folder")
    # 创建新的文件夹
    mkdir -p "./$train_data_dir/$folder_name"
    # 获取随机的num_files个以.json结尾的文件
    files=("$raw_data_dir/$folder_name"/json_"$folder_name"/*.json)
    random_files=($(shuf -e "${files[@]}" -n ${num_files[i]}))
    # 显示进度
    echo 正在处理文件夹 "$raw_data_dir/$folder_name/json_$folder_name" ...
    # 复制文件到新的文件夹
    cp "${random_files[@]}" "./$train_data_dir/$folder_name"
    echo 文件夹 "$raw_data_dir/$folder_name/json_$folder_name" 处理完成。
done