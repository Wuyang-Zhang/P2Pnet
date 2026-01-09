

def generate_file_paths(input_txt_path, output_txt_path, img_folder, txt_folder):
    """
    生成包含.jpg和.txt文件路径的列表，并将它们写入到一个新的文件中。

    :param input_txt_path: 输入的train.txt文件路径
    :param output_txt_path: 输出文件路径
    :param img_folder: 存放.jpg文件的文件夹路径
    :param txt_folder: 存放.txt文件的文件夹路径
    """
    with open(input_txt_path, 'r') as input_file, open(output_txt_path, 'w') as output_file:
        for line in input_file:
            img_name = line.strip()
            img_path = f"{img_folder}/{img_name}.jpg"
            txt_path = f"{txt_folder}/{img_name}.txt"
            output_file.write(f"{img_path} {txt_path}\n")

if __name__ == "__main__":
    input_txt_path = "crowd_datasets/SHHA/train_val_test/val.txt"     # floder train  <--- train.txt  |  floder test <--- val.txt 
    output_txt_path = "crowd_datasets/SHHA/test.txt"  
    img_folder = "crowd_datasets/SHHA/val" 
    txt_folder = "crowd_datasets/SHHA/val" 
    generate_file_paths(input_txt_path, output_txt_path, img_folder, txt_folder)