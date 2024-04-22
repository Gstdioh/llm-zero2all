# 提示过程
## 01
### prompt
```
我的文件大致结构如下：
在./下
- baike2018qa
-- json_baike2018qa
--- 1.json
--- 2.json
--- 3.json
- github-python
-- json_github-python
- news2016zh
-- json_news2016zh
- webtext2019zh
-- json_webtext2019zh
- wikipedia_cn_20240416
-- json_wikipedia_cn_20240416
- wikipedia_en_20220301
-- json_wikipedia_en_20220301
目标：构建新的文件夹./all_train_data，对上述的每个文件夹，从中选出随机2个以.json结尾的文件，复制到新的文件夹中
如对baike2018qa文件夹，选出随机的两个文件如1.json和3.json，将其复制到新的文件夹中，得到如下结果
在./all_train_data下
- baike2018qa
-- json_baike2018qa
--- 1.json
--- 3.json
- github-python
-- json_github-python
- news2016zh
-- json_news2016zh
- webtext2019zh
-- json_webtext2019zh
- wikipedia_cn_20240416
-- json_wikipedia_cn_20240416
- wikipedia_en_20220301
-- json_wikipedia_en_20220301
请用sh脚本实现上述目标
```

### result
```bash
#!/bin/bash

# 创建新的文件夹
mkdir -p ./all_train_data

# 文件夹列表
folders=("./baike2018qa" "./github-python" "./news2016zh" "./webtext2019zh" "./wikipedia_cn_20240416" "./wikipedia_en_20220301")

# 遍历每个文件夹
for folder in folders; do
    # 获取文件夹名称
    folder_name=$(basename "$folder")

    # 创建新的文件夹
    mkdir -p "./all_train_data/$folder_name"

    # 获取随机的两个以.json结尾的文件
    files=("$folder"/*.json)
    random_files=($(shuf -e "${files[@]}" -n 2))

    # 复制文件到新的文件夹
    cp "${random_files[@]}" "./all_train_data/$folder_name"
done
```

## 02
### prompt
```
要选择如./baike2018qa/json_baike2018qa/下的随机num个以.json结尾的文件，然后复制到./all_train_data/baike2018qa/下，注意不同文件夹可以选择不同数量的json文件，所以你需要构建一个列表表示各个文件夹需要选择多少个json文件
```

### result
```bash
#!/bin/bash

# 创建新的文件夹
mkdir -p ./all_train_data

# 文件夹列表
folders=("./baike2018qa" "./github-python" "./news2016zh" "./webtext2019zh" "./wikipedia_cn_20240416" "./wikipedia_en_20220301")

# 遍历每个文件夹
# 定义每个文件夹需要选择的json文件数量
num_files=(2 3 1 2 4 3)

# 遍历每个文件夹
for ((i=0; i<${#folders[@]}; i++)); do
    folder=${folders[i]}
    # 获取文件夹名称
    folder_name=$(basename "$folder")
    # 创建新的文件夹
    mkdir -p "./all_train_data/$folder_name"
    # 获取随机的num_files个以.json结尾的文件
    files=("$folder"/*.json)
    random_files=($(shuf -e "${files[@]}" -n ${num_files[i]}))
    # 复制文件到新的文件夹
    cp "${random_files[@]}" "./all_train_data/$folder_name"
doneone
```

# 修改后的结果
```bash
#!/bin/bash

# 该脚本用于生成训练数据集
# 从多个文件夹中随机选择一定数量的json文件，并将这些文件复制到一个新的文件夹中

# 删除文件夹
if [ -d "./all_train_data" ]; then
    rm -r ./all_train_data
fi

# 创建新的文件夹
mkdir -p ./all_train_data

# 文件夹列表
folders=("./baike2018qa" "./github-python" "./news2016zh" "./webtext2019zh" "./wikipedia_cn_20240416" "./wikipedia_en_20220301")

# 定义每个文件夹需要选择的json文件数量
num_files=(8 11 20 21 14 40)

# 遍历每个文件夹
for ((i=0; i<${#folders[@]}; i++))
do
    folder=${folders[i]}
    # 获取文件夹名称
    folder_name=$(basename "$folder")
    # 创建新的文件夹
    mkdir -p "./all_train_data/$folder_name"
    # 获取随机的num_files个以.json结尾的文件
    files=("$folder"/json_"$folder_name"/*.json)
    random_files=($(shuf -e "${files[@]}" -n ${num_files[i]}))
    # 显示进度
    echo "正在处理文件夹 $folder ..."
    # 复制文件到新的文件夹
    cp "${random_files[@]}" "./all_train_data/$folder_name"
    echo "文件夹 $folder 处理完成。"
done
```
