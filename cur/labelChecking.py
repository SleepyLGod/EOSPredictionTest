import os
import numpy as np

# 配置参数（需要根据实际路径调整）
FEATURE_DIR = "./cur/training_data/ebd/features/llama3_70b"  # 特征文件目录
OUTPUT_FILE = "exceeded_max_len_records.txt"  # 输出文件名

def check_labels():
    # 处理输出路径
    output_dir = os.path.dirname(OUTPUT_FILE) or "."  # 处理空路径情况
    
    # 确保输出目录存在（如果是当前目录会自动跳过）
    os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 遍历特征目录中的所有npz文件
        for filename in os.listdir(FEATURE_DIR):
            if not filename.endswith(".npz"):
                continue
                
            file_path = os.path.join(FEATURE_DIR, filename)
            try:
                # 加载数据文件
                with np.load(file_path, allow_pickle=True) as data:
                    labels = data["labels"]
                    
                    # 检查每个label
                    exceeded_records = []
                    for idx, label in enumerate(labels):
                        f.write(f"Label {idx}: {label}\n")  # 直接写入字典内容
                        # 直接访问字典，无需调用.item()
                        if label.get("over_max_len", False):
                            exceeded_records.append(
                                f"Index: {idx} | Rest Len: {label['rest_len']} | Over Max: {label['over_max_len']}"
                            )
                    
                    # 如果有超过的情况则记录
                    if exceeded_records:
                        f.write(f"File: {filename}\n")
                        f.write(f"Total Labels: {len(labels)}\n")
                        f.write("Exceeded Records:\n")
                        f.write("\n".join(exceeded_records))
                        f.write("\n\n" + "="*80 + "\n\n")
                        
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == "__main__":
    check_labels()
    print(f"检测完成，结果已保存至: {OUTPUT_FILE}")