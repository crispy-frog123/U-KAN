import os
import numpy as np
import h5py
import scipy.io as sio
from tqdm import tqdm


def load_mat_flexible(path, var_name=None):
    """支持 v7.3 (HDF5) 和 旧版 .mat 格式"""
    try:
        data = sio.loadmat(path)
        available_vars = [k for k in data.keys() if not k.startswith('__')]
        if var_name not in available_vars:
            print(f"⚠️ 文件 {os.path.basename(path)} 中未找到 '{var_name}'。自动使用: {available_vars[0]}")
            var_name = available_vars[0]
        return data[var_name]
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            available_vars = list(f.keys())
            if var_name not in available_vars:
                # 过滤掉 HDF5 自动生成的元数据 key
                valid_vars = [v for v in available_vars if v != '#refs#']
                print(f"⚠️ 文件 {os.path.basename(path)} (v7.3) 中未找到 '{var_name}'。")
                print(f"   可用变量: {valid_vars}")
                var_name = valid_vars[0]

            data = f[var_name][:]
            # 维度调整：确保第 0 维是样本数 N
            if data.ndim == 2 and data.shape[0] < data.shape[1]:
                data = data.T
            return data


def split_mat_to_npy(config):
    # 文件夹重命名为 labels
    out_img_dir = os.path.join(config['out_dir'], 'images')
    out_label_dir = os.path.join(config['out_dir'], 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    print("正在加载数据...")
    r_img = load_mat_flexible(config['real_img_path'], config['img_var'])
    i_img = load_mat_flexible(config['imag_img_path'], config['imag_var'])
    r_label = load_mat_flexible(config['real_label_path'], config['label_var'])
    i_label = load_mat_flexible(config['imag_label_path'], config['imag_label_var'])

    num_samples = r_img.shape[0]
    H, W = config['size']

    for i in tqdm(range(num_samples), desc="拆分并保存"):
        # 图像合并 (H, W, 2)
        img_sample = np.stack([
            r_img[i].reshape(H, W),
            i_img[i].reshape(H, W)
        ], axis=-1).astype('float32')

        # 标签合并 (H, W, 2)
        label_sample = np.stack([
            r_label[i].reshape(H, W),
            i_label[i].reshape(H, W)
        ], axis=-1).astype('float32')

        np.save(os.path.join(out_img_dir, f"{i}.npy"), img_sample)
        np.save(os.path.join(out_label_dir, f"{i}.npy"), label_sample)


# --- 请根据你的实际情况微调这里的变量名 ---
config = {
    'real_img_path': 'inputs/input_new/chi0_all_real_mnist.mat',
    'imag_img_path': 'inputs/input_new/chi0_all_imag_mnist.mat',
    'real_label_path': 'inputs/label_new/chi_all_real_mnist.mat',
    'imag_label_path': 'inputs/label_new/chi_all_imag_mnist.mat',

    'img_var': 'chi0_all_real',
    'imag_var': 'chi0_all_imag',
    'label_var': 'label',  # <--- 确认为 'label'
    'imag_label_var': 'label',  # <--- 如果虚部文件的变量名也叫 'label'

    'size': (64, 64),
    'out_dir': 'inputs/processed_data'
}

if __name__ == '__main__':
    split_mat_to_npy(config)