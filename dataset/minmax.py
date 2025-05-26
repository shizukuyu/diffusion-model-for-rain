import numpy as np
import os
from glob import glob

def calculate_and_save_stats(lr_dir, output_max_path, output_min_path):
    """
    计算LR目录下所有.npy文件中各变量的最大最小值，并保存为npy文件
    
    参数:
    lr_dir: LR数据所在目录
    output_max_path: 最大值保存路径
    output_min_path: 最小值保存路径
    """
    # 变量名称和对应的索引
    var_names = ['u', 'v', 't', 'r']
    var_indices = {name: i for i, name in enumerate(var_names)}
    
    # 初始化最大最小值数组
    max_values = np.full((len(var_names),), -np.inf)
    min_values = np.full((len(var_names),), np.inf)
    
    # 获取所有LR数据文件
    lr_files = glob(os.path.join(lr_dir, "*.npy"))
    print(f"找到{len(lr_files)}个LR数据文件")
    
    # 遍历所有文件计算统计量
    for file_path in lr_files:
        try:
            data = np.load(file_path)
            
            # 确保数据是3D [C, H, W]
            if data.ndim == 2:
                data = np.expand_dims(data, axis=0)
            
            # 检查通道数是否符合预期
            if data.shape[0] < len(var_names):
                print(f"警告: 文件 {file_path} 的通道数少于预期({len(var_names)})")
            
            # 更新每个变量的最大最小值
            for var_name, idx in var_indices.items():
                if idx < data.shape[0]:  # 确保索引不越界
                    var_data = data[idx]
                    var_max = np.max(var_data)
                    var_min = np.min(var_data)
                    
                    if var_max > max_values[idx]:
                        max_values[idx] = var_max
                    if var_min < min_values[idx]:
                        min_values[idx] = var_min
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 保存统计量
    np.save(output_max_path, max_values)
    np.save(output_min_path, min_values)
    
    print(f"最大值统计量已保存到: {output_max_path}")
    print(f"最小值统计量已保存到: {output_min_path}")
    print("统计量详情:")
    for i, var_name in enumerate(var_names):
        print(f"{var_name}: min={min_values[i]:.4f}, max={max_values[i]:.4f}")
    
    return max_values, min_values

# 使用示例
if __name__ == "__main__":
    lr_dir = "/workspace3/suwen/ddpm/dataset/lr/"
    output_max_path = "/workspace3/suwen/ddpm/dataset/lr/max_values.npy"
    output_min_path = "/workspace3/suwen/ddpm/dataset/lr/min_values.npy"
    
    calculate_and_save_stats(lr_dir, output_max_path, output_min_path)    