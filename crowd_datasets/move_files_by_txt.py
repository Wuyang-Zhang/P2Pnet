import os
import shutil

def move_files_by_txt(txt_folder, source_folder, target_root_folder, file_exts=[".jpg", ".txt"]):
    """
    核心函数：根据txt文件列表，批量移动对应扩展名的文件到指定文件夹
    
    函数名称：move_files_by_txt
    作用：
        1. 读取txt_folder下的train.txt/val.txt/test.txt文件
        2. 按文件中的文件名，从source_folder移动指定扩展名（默认jpg和txt）的文件
        3. 自动创建train/val/test目标文件夹，将文件分类移动进去
    参数：
        txt_folder (str): 存放train/val/test.txt的文件夹路径（如crowd_datasets\SHHA\train_val_test）
        source_folder (str): 存放待移动的图片和txt文件的源文件夹
        target_root_folder (str): 目标根文件夹（会在该目录下创建train/val/test子文件夹）
        file_exts (list): 需要移动的文件扩展名列表，默认[".jpg", ".txt"]
    返回值：
        None（直接打印移动日志）
    """
    # 定义txt文件与目标文件夹的映射关系
    txt_to_folder = {
        "train.txt": "train",
        "val.txt": "val",
        "test.txt": "test"
    }
    
    # 遍历每个txt文件，处理对应文件的移动
    for txt_filename, target_subfolder in txt_to_folder.items():
        # 拼接txt文件的完整路径
        txt_file_path = os.path.join(txt_folder, txt_filename)
        if not os.path.exists(txt_file_path):
            print(f"⚠️  未找到{txt_filename}，跳过该分组移动")
            continue
        
        # 拼接目标子文件夹路径（如target_root_folder/train）
        target_folder = os.path.join(target_root_folder, target_subfolder)
        # 自动创建目标文件夹（不存在则创建）
        os.makedirs(target_folder, exist_ok=True)
        
        # 统计移动结果
        moved_count = 0
        not_found_count = 0
        
        # 读取txt文件中的文件名列表
        with open(txt_file_path, "r", encoding="utf-8") as f:
            for line_num, filename_prefix in enumerate(f, 1):
                filename_prefix = filename_prefix.strip()
                if not filename_prefix:  # 跳过空行
                    continue
                
                # 遍历需要移动的文件扩展名（jpg和txt）
                for ext in file_exts:
                    # 拼接源文件完整路径
                    source_file_path = os.path.join(source_folder, filename_prefix + ext)
                    # 拼接目标文件完整路径
                    target_file_path = os.path.join(target_folder, filename_prefix + ext)
                    
                    # 检查源文件是否存在
                    if os.path.exists(source_file_path):
                        try:
                            # 移动文件（覆盖已存在的同名文件）
                            shutil.move(source_file_path, target_file_path)
                            moved_count += 1
                        except Exception as e:
                            print(f"❌ 移动失败 [{txt_filename} 第{line_num}行]: {filename_prefix}{ext} - {e}")
                            not_found_count += 1
                    else:
                        print(f"❌ 文件未找到 [{txt_filename} 第{line_num}行]: {filename_prefix}{ext}")
                        not_found_count += 1
        
        # 打印该分组的移动结果
        print(f"\n📊 {target_subfolder} 分组移动完成：")
        print(f"   成功移动：{moved_count} 个文件")
        print(f"   失败/未找到：{not_found_count} 个文件")

# ------------------------------ 辅助函数（可选） ------------------------------
def batch_move_train_val_test():
    """
    快捷函数：batch_move_train_val_test
    作用：
        封装固定路径，一键执行文件移动（简化调用，无需重复传参）
    使用场景：
        路径固定时，直接调用该函数即可，无需手动传参
    """
    # 请根据你的实际路径修改以下参数
    txt_folder = "crowd_datasets/SHHA/train_val_test"  # 存放train/val/test.txt的文件夹
    source_folder = "crowd_datasets/SHHA/ALL_IMG"  # 存放图片和txt的源文件夹
    target_root_folder = "crowd_datasets/SHHA"  # 目标根文件夹（会创建train/val/test子文件夹）
    
    # 调用核心移动函数
    move_files_by_txt(txt_folder, source_folder, target_root_folder)

 
if __name__ == "__main__":
    # 方式1：直接调用快捷函数（推荐，路径已封装）
    batch_move_train_val_test()
    
    # 方式2：手动传参调用核心函数（灵活调整路径）
    # move_files_by_txt(
    #     txt_folder=r"你的train/val/test.txt文件夹路径",
    #     source_folder=r"你的图片和txt源文件夹路径",
    #     target_root_folder=r"目标根文件夹路径"
    # )