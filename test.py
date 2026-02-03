import argparse
import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
import matplotlib.pyplot as plt

# 导入项目模块
import archs
from dataset_e import ComplexMatDataset, TransformSubset
# 确保 utils.py 里已经添加了 calc_relative_error
from utils import AverageMeter, calc_relative_error

# 导入高级评价指标库
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func


def parse_args():
    parser = argparse.ArgumentParser()
    # 指定 output 文件夹路径
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory (e.g., outputs/2000_[...])')
    parser.add_argument('--checkpoint', type=str, default='model.pth',
                        help='Model filename to load (default: model.pth which is the best)')
    parser.add_argument('--save_visuals', action='store_true',
                        help='Save predicted images')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 Config
    config_path = os.path.join(args.exp_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Testing experiment: {config.get('name', 'Unknown')}")
    print(f" Device: {device}")

    # 2. 准备数据集 (必须与 train.py 使用完全相同的种子和逻辑)
    print("Loading Dataset...")
    val_transform = Compose([Resize(config['input_h'], config['input_w'])])

    # 构建路径辅助函数
    def build_path(base_dir, file_path):
        if os.path.isabs(file_path): return file_path
        return os.path.normpath(os.path.join(base_dir, file_path))

    full_dataset = ComplexMatDataset(
        real_img_path=build_path(config['data_dir'], config['real_img_file']),
        imag_img_path=build_path(config['data_dir'], config['imag_img_file']),
        real_label_path=build_path(config['data_dir'], config['real_label_file']),
        imag_label_path=build_path(config['data_dir'], config['imag_label_file']),
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,  # 测试集不做翻转增强
        normalize_method='z-score'
    )

    # === 复现 8:1:1 划分逻辑 (关键！) ===
    indices = list(range(len(full_dataset)))
    np.random.seed(config['dataseed'])  # 必须与 train.py 种子一致
    np.random.shuffle(indices)

    test_split = int(np.floor(0.1 * len(full_dataset)))
    test_indices = indices[:test_split]  # 取前 10% 作为测试集 (0-199)

    test_dataset = TransformSubset(full_dataset, test_indices, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch=1 方便逐个计算

    print(f"Test Set Size: {len(test_dataset)} samples (Independent from Train/Val)")

    # 3. 加载模型
    print("Loading Model...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan']
    ).to(device)

    model_path = os.path.join(args.exp_dir, args.checkpoint)
    print(f"Loading weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 开始测试
    # 定义指标记录器
    metrics = {
        'mse': AverageMeter(),
        'psnr': AverageMeter(),
        'ssim': AverageMeter(),
        're_real': AverageMeter(),  # [新增] 实部相对误差
        're_imag': AverageMeter(),  # [新增] 虚部相对误差
        're_avg': AverageMeter()  # [新增] 平均相对误差
    }

    save_dir = os.path.join(args.exp_dir, 'test_results_final')
    os.makedirs(save_dir, exist_ok=True)

    print("\nStarting Inference...")
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(tqdm(test_loader)):
            input = input.to(device)
            target = target.to(device)

            # 推理
            if config['deep_supervision']:
                outputs = model(input)
                pred = outputs[-1]
            else:
                pred = model(input)

            # --- 数据准备 ---
            # shape: (2, H, W)
            pred_np = pred.cpu().numpy().squeeze()
            target_np = target.cpu().numpy().squeeze()

            # --- 1. 基础指标 (MSE) ---
            mse = np.mean((pred_np - target_np) ** 2)
            metrics['mse'].update(mse)

            # --- 2. 相对误差 (Relative Error) - 对应 DeepNIS RMSE ---
            # 分别计算实部(通道0)和虚部(通道1)
            re_real = calc_relative_error(pred_np[0], target_np[0])
            re_imag = calc_relative_error(pred_np[1], target_np[1])
            re_avg = (re_real + re_imag) / 2.0

            metrics['re_real'].update(re_real)
            metrics['re_imag'].update(re_imag)
            metrics['re_avg'].update(re_avg)

            # --- 3. 图像指标 (PSNR & SSIM) ---
            # 动态获取 range (Z-score 后的数据范围通常在 -3 到 3 之间)
            d_range = target_np.max() - target_np.min()

            psnr_val = psnr_func(target_np, pred_np, data_range=d_range)
            # 1. 将 (C, H, W) 转置为 (H, W, C)
            target_np_HWC = target_np.transpose(1, 2, 0)
            pred_np_HWC = pred_np.transpose(1, 2, 0)

            # 2. 使用 multichannel=True (旧版参数)
            ssim_val = ssim_func(target_np_HWC, pred_np_HWC, data_range=d_range, multichannel=True)

            metrics['psnr'].update(psnr_val)
            metrics['ssim'].update(ssim_val)

            # --- 可视化保存 (前10张) ---
            if args.save_visuals and i < 10:
                fig, ax = plt.subplots(1, 4, figsize=(12, 3))
                # Real GT
                ax[0].imshow(target_np[0], cmap='jet')
                ax[0].set_title('GT Real')
                ax[0].axis('off')
                # Real Pred
                ax[1].imshow(pred_np[0], cmap='jet')
                ax[1].set_title(f'Pred Real\nRE: {re_real:.4f}')
                ax[1].axis('off')
                # Imag GT
                ax[2].imshow(target_np[1], cmap='jet')
                ax[2].set_title('GT Imag')
                ax[2].axis('off')
                # Imag Pred
                ax[3].imshow(pred_np[1], cmap='jet')
                ax[3].set_title(f'Pred Imag\nRE: {re_imag:.4f}')
                ax[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'test_{i}_RE_{re_avg:.3f}.png'))
                plt.close()

    # 5. 输出最终报告
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS (Comparison with DeepNIS)")
    print("=" * 50)
    print(f"Total Samples: {len(test_dataset)}")
    print("-" * 30)
    print(f"Avg MSE:            {metrics['mse'].avg:.6f}")
    print(f"Avg PSNR:           {metrics['psnr'].avg:.4f} dB")
    print(f"Avg SSIM:           {metrics['ssim'].avg:.4f}  (Target: >0.90)")
    print("-" * 30)
    print("Relative Error (RE) / DeepNIS RMSE:")
    print(f"  Real Part:        {metrics['re_real'].avg:.4f}  (Target: <0.06)")
    print(f"  Imag Part:        {metrics['re_imag'].avg:.4f}")
    print(f"  Average:          {metrics['re_avg'].avg:.4f}")
    print("=" * 50)

    # 保存到 txt
    with open(os.path.join(args.exp_dir, 'final_metrics_report.txt'), 'w') as f:
        f.write("Method: U-KAN (Test Set Evaluation)\n")
        f.write("=" * 40 + "\n")
        f.write(f"Avg SSIM:           {metrics['ssim'].avg:.4f}\n")
        f.write(f"Relative Error:     {metrics['re_avg'].avg:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"RE (Real):          {metrics['re_real'].avg:.4f}\n")
        f.write(f"RE (Imag):          {metrics['re_imag'].avg:.4f}\n")
        f.write(f"Avg PSNR:           {metrics['psnr'].avg:.4f}\n")
        f.write(f"Avg MSE:            {metrics['mse'].avg:.6f}\n")


if __name__ == '__main__':
    main()