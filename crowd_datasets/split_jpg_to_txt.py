import os
import random

def split_jpg_to_txt(jpg_folder_path, output_txt_folder, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    将指定文件夹中的jpg文件按6:2:2比例划分，写入train/val/test.txt（输出到指定文件夹）
    
    Args:
        jpg_folder_path: 存放jpg文件的源文件夹路径
        output_txt_folder: 生成的txt文件要保存的目标文件夹（不存在则自动创建）
        train_ratio/val_ratio/test_ratio: 划分比例，默认6:2:2
    """
    # 1. 检查jpg源文件夹是否存在
    if not os.path.exists(jpg_folder_path):
        print(f"错误：存放jpg的文件夹不存在 - {jpg_folder_path}")
        return
    
    # 2. 创建txt输出文件夹（不存在则自动创建）
    os.makedirs(output_txt_folder, exist_ok=True)
    print(f"输出文件夹已准备好：{output_txt_folder}")
    
    # 3. 筛选所有.jpg文件，并提取前缀名称（去掉.jpg后缀）
    jpg_files = []
    for filename in os.listdir(jpg_folder_path):
        if filename.lower().endswith(".jpg"):  # 兼容大写JPG后缀
            # 提取前缀（去掉.jpg后缀）
            prefix = os.path.splitext(filename)[0]
            jpg_files.append(prefix)
    
    if not jpg_files:
        print("未找到任何.jpg文件，请检查源文件夹路径是否正确")
        return
    
    # 4. 随机打乱文件列表（保证划分随机性）
    random.shuffle(jpg_files)
    total = len(jpg_files)
    print(f"共找到 {total} 个jpg文件，开始按6:2:2比例划分...")
    
    # 5. 计算各分组的数量（避免小数误差，test取剩余部分）
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    
    # 6. 划分分组
    train_list = jpg_files[:train_num]
    val_list = jpg_files[train_num:train_num+val_num]
    test_list = jpg_files[train_num+val_num:]
    
    # 7. 将分组写入指定文件夹下的txt文件
    def write_to_txt(data_list, txt_name):
        # 拼接txt文件的完整路径
        txt_path = os.path.join(output_txt_folder, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            for name in data_list:
                f.write(f"{name}\n")
        print(f"✅ 已写入 {len(data_list)} 个文件名到 {txt_name}（路径：{txt_path}）")
    
    # 写入三个txt文件
    write_to_txt(train_list, "train.txt")
    write_to_txt(val_list, "val.txt")
    write_to_txt(test_list, "test.txt")
    
    # 打印汇总信息
    print("\n=== 划分完成 ===")
    print(f"总文件数：{total}")
    print(f"Train：{len(train_list)} 个（占比 {len(train_list)/total:.2%}）")
    print(f"Val：{len(val_list)} 个（占比 {len(val_list)/total:.2%}）")
    print(f"Test：{len(test_list)} 个（占比 {len(test_list)/total:.2%}）")

if __name__ == "__main__":

    jpg_folder = "crowd_datasets/SHHA/ALL_IMG"   
    output_txt_folder = "crowd_datasets/SHHA/train_val_test"
    # =======================================================
    split_jpg_to_txt(jpg_folder, output_txt_folder)
