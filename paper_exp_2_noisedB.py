import argparse
import os
import heapq
import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
from skimage.metrics import structural_similarity as ssim_func

import archs
from dataset_e import ComplexMatDataset, TransformSubset
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--checkpoint', type=str, default='model.pth', help='Model checkpoint')
    parser.add_argument('--save_dir', type=str, default='paper_results/dim2_snr_final_multi', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Inference batch size')
    return parser.parse_args()


def add_noise_snr(img_tensor, snr_db):
    """
    闂佸搫绉烽～澶婄暤?SNR (dB) 濠电儑缍€椤曆勬叏閻愭惌娈楁俊顖滅帛閻掑潡鏌ｈ濡嫬鈻嶉弮鈧粩?
    """
    if snr_db is None: return img_tensor  # Clean case

    device = img_tensor.device
    B, C, H, W = img_tensor.shape
    noisy_imgs = []

    for i in range(B):
        img = img_tensor[i]
        signal_std = torch.std(img)
        noise_std = signal_std / (10 ** (snr_db / 20.0))
        noise = torch.randn_like(img) * noise_std
        noisy_imgs.append(img + noise)

    return torch.stack(noisy_imgs).to(device)


def calculate_batch_ssim(pred, target):
    ssim_scores = []
    B = pred.shape[0]
    for i in range(B):
        p = pred[i]
        t = target[i]
        range_r = t[0].max() - t[0].min() + 1e-6
        s_r = ssim_func(t[0], p[0], data_range=range_r)
        range_i = t[1].max() - t[1].min() + 1e-6
        s_i = ssim_func(t[1], p[1], data_range=range_i)
        ssim_scores.append((s_r + s_i) / 2.0)
    return ssim_scores


def run_noise_test(model, loader, device, snr_levels, num_vis_samples=5):
    results = {}

    vis_keys = list(snr_levels)
    top_k = max(1, int(num_vis_samples))
    candidate_pool_k = max(top_k * 20, top_k)
    top_heap = []  # min-heap: (ssim, sample_index)
    top_clean_data = {}  # sample_index -> data at clean
    top_indices_ordered = []
    top_index_set = set()
    vis_data_by_sample = {}

    def target_feature(sample_data):
        target = sample_data['target']
        feat = np.concatenate([
            target[0].reshape(-1),
            np.abs(target[1]).reshape(-1)
        ]).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    for snr in snr_levels:
        label = "Clean" if snr is None else f"{snr}dB"
        print(f"Testing with SNR = {label} ...")
        all_ssims = []
        global_idx = 0

        with torch.no_grad():
            for input_tensor, target, _ in tqdm(loader, desc=f"SNR {label}", leave=False):
                input_tensor = input_tensor.to(device)
                target = target.to(device)

                noisy_input = add_noise_snr(input_tensor, snr)

                if hasattr(model, 'deep_supervision') and model.deep_supervision:
                    output = model(noisy_input)
                    if isinstance(output, list):
                        output = output[-1]
                else:
                    output = model(noisy_input)

                pred_np = output.cpu().numpy()
                target_np = target.cpu().numpy()
                batch_ssims = calculate_batch_ssim(pred_np, target_np)
                all_ssims.extend(batch_ssims)

                batch_size = input_tensor.shape[0]

                if snr is None and (None in vis_keys):
                    for b in range(batch_size):
                        sample_idx = global_idx + b
                        sample_ssim = batch_ssims[b]
                        sample_data = {
                            'input': noisy_input[b].cpu().numpy(),
                            'target': target[b].cpu().numpy(),
                            'pred': output[b].cpu().numpy()
                        }

                        if len(top_heap) < candidate_pool_k:
                            heapq.heappush(top_heap, (sample_ssim, sample_idx))
                            top_clean_data[sample_idx] = sample_data
                        elif sample_ssim > top_heap[0][0]:
                            _, removed_idx = heapq.heapreplace(top_heap, (sample_ssim, sample_idx))
                            top_clean_data.pop(removed_idx, None)
                            top_clean_data[sample_idx] = sample_data

                elif snr in vis_keys and top_index_set:
                    for b in range(batch_size):
                        sample_idx = global_idx + b
                        if sample_idx in top_index_set:
                            vis_data_by_sample[sample_idx][snr] = {
                                'input': noisy_input[b].cpu().numpy(),
                                'target': target[b].cpu().numpy(),
                                'pred': output[b].cpu().numpy()
                            }

                global_idx += batch_size

        results[snr] = np.mean(all_ssims)
        print(f"   -> Mean SSIM: {results[snr]:.4f}")

        if snr is None and top_heap:
            # Greedy diversity selection from high-SSIM candidate pool.
            top_pairs = sorted(top_heap, key=lambda x: x[0], reverse=True)
            selected = []
            selected_feats = []
            max_similarity = 0.88

            for score, idx in top_pairs:
                feat = target_feature(top_clean_data[idx])
                if not selected_feats:
                    selected.append((score, idx))
                    selected_feats.append(feat)
                else:
                    sim_to_selected = [float(np.dot(feat, f)) for f in selected_feats]
                    if max(sim_to_selected) < max_similarity:
                        selected.append((score, idx))
                        selected_feats.append(feat)
                if len(selected) >= top_k:
                    break

            # Fallback: if diversity constraint is strict, fill remaining by SSIM rank.
            if len(selected) < top_k:
                chosen = {idx for _, idx in selected}
                for score, idx in top_pairs:
                    if idx not in chosen:
                        selected.append((score, idx))
                        chosen.add(idx)
                    if len(selected) >= top_k:
                        break

            top_indices_ordered = [idx for _, idx in selected]
            top_index_set = set(top_indices_ordered)
            vis_data_by_sample = {idx: {None: top_clean_data[idx]} for idx in top_indices_ordered}
            print(f"   -> Selected top-{len(top_indices_ordered)} diverse samples by clean SSIM for visualization.")

    vis_data_list = [vis_data_by_sample[idx] for idx in top_indices_ordered if idx in vis_data_by_sample]
    return results, vis_data_list


def plot_noise_curve_ieee(results, save_dir):
    """
    缂傚倷鐒﹂敋闁?IEEE 婵＄偑鍊涘▔娑㈠垂閺冣偓椤︾増鎯旈姀銏狀棔闂佺鍩栭…鍥吹鎼淬劌鐐?
    """
    # ------------------- IEEE 婵＄偛顑呯€涒晠鎮ч崫銉﹀闁告挆鍛€?-------------------
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    # ----------------------------------------------------

    plt.figure(figsize=(8, 6))

    # 1. 闂佸湱绮崝鏇°亹閸ヮ剙鏋侀柣妤€鐗嗙粊?
    snr_keys = [k for k in results.keys() if k is not None]
    snr_keys.sort(reverse=True)  # 闂傚倸瀚粔瀵歌姳椤撱垹绠抽柟鐑樺灥閻? [40, 39, ..., 10]

    ssims = [results[k] for k in snr_keys]
    clean_ssim = results[None]

    # 2. 缂傚倷鐒﹂敋闁?Clean 闂佸搫绉村ú銈夊闯椤栨粎妫?
    plt.axhline(y=clean_ssim, color='gray', linestyle='--', linewidth=1.5,
                label=f'Clean Baseline (SSIM={clean_ssim:.3f})', zorder=1)

    # 3. 缂傚倷鐒﹂敋闁糕晜顨婇獮鈧憸鎴﹀礂濮椻偓楠炲骸螣閻撳孩鐤?
    plt.plot(snr_keys, ssims, marker='o', markersize=6, linewidth=2.0,
             color='#d62728', label='UKAN Robustness', zorder=2)

    # 4. 闂佺鍕闁绘牭绲惧顏嗗鐎ｉ潧鏅紓?
    plt.xlabel('SNR (dB)', fontsize=16, fontweight='bold')
    plt.ylabel('Mean SSIM', fontsize=16, fontweight='bold')

    # X 闁哄鍋炲娆撳春閵忊剝鍎熼柨鏃囧亹缁愭鏌涘▎蹇曅㈤柟顔界矒瀵即宕滆娴?(40 -> 10)
    plt.xlim(42, 8)
    plt.xticks(np.arange(10, 45, 5))

    # Y 闁哄鍋熼ˉ鎰偓鍨墵瀹曞墎澹曠€ｎ亝顔囬梻渚囧亜閸婂摜鑺?
    min_y = min(ssims)
    max_y = max(clean_ssim, max(ssims))
    plt.ylim(min_y - 0.05, max_y + 0.02)

    # 5. 闂佹悶鍎辫ぐ鐐垫?
    plt.legend(loc='lower left', frameon=True, edgecolor='black', fancybox=False, fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'snr_ssim_curve_ieee.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"闂?Curve saved to: {save_path}")


def plot_visual_matrix_paper_style(vis_data, save_dir, sample_id=0):
    """
    闁告瑯鍨甸～瀣礌閺嶎偆鍙愰梻?Cols: [Ground Truth, Clean, 40dB, 30dB, 20dB, 10dB]
    GT 闁告帗銇炵粭澶愬及閸撗佷粵閺夊牊鎸搁崣鍡涙晬鐏炶偐鐭岄柡鍕⒔閵?GT 閺夊牊鎸搁崵顓㈡晬鐏炲€熷珯闁衡偓閹勮含闁?GT 閺夊牊鎸搁崣?閺夊牊鎸搁崵顓㈡儍閸曨亣鍘梻鍌滅節缂嶅懐绱旈琛″亾?    """
    col_keys = [None, 40, 30, 20, 10]
    col_keys = [k for k in col_keys if k in vis_data]

    cols = len(col_keys) + 1  # +1 for Ground Truth
    rows = 4

    row_labels = [
        "Scattered E (Real)",
        "Chi (Real)",
        "Scattered E (|Imag|)",
        "Chi (|Imag|)"
    ]

    if None in vis_data:
        gt_source = vis_data[None]
    else:
        gt_source = vis_data[col_keys[0]]

    gt_target = gt_source['target']
    gt_target_r = gt_source['target'][0]
    gt_target_i = np.abs(gt_source['target'][1])

    def sample_ssim(pred_arr, target_arr):
        range_r = target_arr[0].max() - target_arr[0].min() + 1e-6
        s_r = ssim_func(target_arr[0], pred_arr[0], data_range=range_r)
        range_i = target_arr[1].max() - target_arr[1].min() + 1e-6
        s_i = ssim_func(target_arr[1], pred_arr[1], data_range=range_i)
        return (s_r + s_i) / 2.0

    # Unified ranges for fair comparison
    all_pred_r, all_pred_i, all_input_r, all_input_i = [], [], [], []
    for k in col_keys:
        all_pred_r.append(vis_data[k]['pred'][0])
        all_pred_i.append(np.abs(vis_data[k]['pred'][1]))
        all_input_r.append(vis_data[k]['input'][0])
        all_input_i.append(np.abs(vis_data[k]['input'][1]))

    input_r_range = (
        min([x.min() for x in all_input_r]),
        max([x.max() for x in all_input_r])
    )
    output_r_range = (
        min(gt_target_r.min(), min([x.min() for x in all_pred_r])),
        max(gt_target_r.max(), max([x.max() for x in all_pred_r]))
    )
    input_i_range = (0, max([x.max() for x in all_input_i]))
    output_i_range = (0, max(gt_target_i.max(), max([x.max() for x in all_pred_i])))
    ranges = [input_r_range, output_r_range, input_i_range, output_i_range]

    # Add a dedicated narrow column for colorbars to keep all image columns same size.
    fig = plt.figure(figsize=(2.5 * cols + 0.6, 2.5 * rows))
    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.08], wspace=0.05, hspace=0.05)

    # GT column: show output only and center across two rows
    ax_gt_r = fig.add_subplot(gs[0:2, 0])
    ax_gt_r.imshow(gt_target_r, cmap='jet', vmin=output_r_range[0], vmax=output_r_range[1])
    ax_gt_r.set_xticks([])
    ax_gt_r.set_yticks([])

    # Use a transparent title axis on row 0 to align title height with Clean column.
    ax_gt_title = fig.add_subplot(gs[0, 0], frameon=False)
    ax_gt_title.patch.set_alpha(0.0)
    ax_gt_title.set_xticks([])
    ax_gt_title.set_yticks([])
    for spine in ax_gt_title.spines.values():
        spine.set_visible(False)
    ax_gt_title.set_title("Ground Truth", fontsize=14, fontweight='bold', pad=8, fontname='Times New Roman')

    ax_gt_i = fig.add_subplot(gs[2:4, 0])
    ax_gt_i.imshow(gt_target_i, cmap='jet', vmin=output_i_range[0], vmax=output_i_range[1])
    ax_gt_i.set_xticks([])
    ax_gt_i.set_yticks([])
    ax_gt_i.set_xlabel("SSIM = 1.0000", fontsize=13, fontweight='bold', labelpad=10, fontname='Times New Roman')

    # Other columns keep input/prediction comparison
    for c, k in enumerate(col_keys, start=1):
        data = vis_data[k]
        col_title = "Clean" if k is None else f"{k}dB"
        ssim_val = sample_ssim(data['pred'], gt_target)
        imgs = [
            data['input'][0],
            data['pred'][0],
            np.abs(data['input'][1]),
            np.abs(data['pred'][1])
        ]

        for r in range(rows):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(imgs[r], cmap='jet', vmin=ranges[r][0], vmax=ranges[r][1])
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold', pad=8, fontname='Times New Roman')

            if c == cols - 1:
                cax = fig.add_subplot(gs[r, cols])
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(row_labels[r], size=16, weight='bold', fontname='Times New Roman')
                cbar.ax.tick_params(labelsize=13, width=1.5)
                for tick in cbar.ax.get_yticklabels():
                    tick.set_fontweight('bold')
                    tick.set_fontname('Times New Roman')

            if r == 3:
                ax.set_xlabel(f"SSIM = {ssim_val:.4f}",
                              fontsize=13,
                              fontweight='bold',
                              labelpad=10,
                              fontname='Times New Roman')

    save_path = os.path.join(save_dir, f'snr_vis_paper_style_sample{sample_id + 1}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization for Sample {sample_id + 1} saved to: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Config
    config_path = os.path.join(args.exp_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    def build_path(base_dir, file_path):
        if os.path.isabs(file_path): return file_path
        return os.path.normpath(os.path.join(base_dir, file_path))

    full_dataset = ComplexMatDataset(
        real_img_path=build_path(config['data_dir'], config['real_img_file']),
        imag_img_path=build_path(config['data_dir'], config['imag_img_file']),
        real_label_path=build_path(config['data_dir'], config['real_label_file']),
        imag_label_path=build_path(config['data_dir'], config['imag_label_file']),
        img_size=(config['original_img_size'], config['original_img_size']),
        normalize_method='z-score'
    )
    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path): full_dataset.load_stats(stats_path)

    indices = list(range(len(full_dataset)))
    np.random.seed(config['dataseed'])
    np.random.shuffle(indices)
    test_indices = indices[:200]

    val_transform = Compose([Resize(config['input_h'], config['input_w'])])
    test_set = TransformSubset(full_dataset, test_indices, val_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print(f"Loading Model: {config['arch']}...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False),
        use_edge_residual_refine=config.get('use_edge_residual_refine', False),
        use_edge_multiscale=config.get('use_edge_multiscale', False),
        use_edge_sparse_focus=config.get('use_edge_sparse_focus', False),
        edge_focus_tau=config.get('edge_focus_tau', 0.25),
        edge_focus_gamma=config.get('edge_focus_gamma', 2.0),
        use_edge_center_boost=config.get('use_edge_center_boost', False),
        edge_center_boost=config.get('edge_center_boost', 0.1),
        edge_center_sigma=config.get('edge_center_sigma', 0.1),
        edge_refine_scale=config.get('edge_refine_scale', 1.0),
        edge_refine_mid=config.get('edge_refine_mid', 64),
        use_detail_skip_refine=config.get('use_detail_skip_refine', False),
        detail_refine_scale=config.get('detail_refine_scale', 1.0),
        detail_refine_mid=config.get('detail_refine_mid', 64),
        use_fourier_refine=config.get('use_fourier_refine', False),
        fourier_use_fft=config.get('fourier_use_fft', False),
        fourier_refine_scale=config.get('fourier_refine_scale', 1.0),
        fourier_refine_mid=config.get('fourier_refine_mid', 64)
    ).to(device)
    ckpt = torch.load(os.path.join(args.exp_dir, args.checkpoint), map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Run Experiment
    # Only evaluate clean / 40 / 30 / 20 / 10 dB
    snr_levels = [None, 40, 30, 20, 10]

    # [婵烇絽娴傞崰妤呭极缁?闁荤姳绀佹晶浠嬫偪閸℃瑦鍟哄〒姘ｅ亾闁逞屽墮椤︻垵銇愰崶顒佸剭闁告洦鍋勭拋鏌ユ偡濞嗗繑顥滈悗鍨叀瀵粙鏌ㄧ€ｎ偅瀚崇紓鍌欑鐎氼參寮?
    NUM_VIS_SAMPLES = 10
    print(f"Starting SNR Noise Test ({len(snr_levels)} levels), collecting {NUM_VIS_SAMPLES} visual samples...")
    results, vis_data_list = run_noise_test(model, test_loader, device, snr_levels, num_vis_samples=NUM_VIS_SAMPLES)

    # Skip curve plotting, keep table + visualization only

    # [婵烇絽娴傞崰妤呭极缁?閻庣敻鍋婃禍鐐虹嵁閸℃瑧纾兼俊顖氭惈閻?5 閻庢鍠氭慨纾嬨亹閼碱剚鍠嗛柛鈩冾殔椤曆囨煕?
    print(f"Generating {len(vis_data_list)} visualization matrices...")
    for i, vis_data in enumerate(vis_data_list):
        if vis_data:  # 缂佺虎鍙庨崰鏇犳崲濮樻墎鍋撳☉娆樻當闁告埃鍋撴繛鎴炴尭缁夊磭鎷归悢铏圭煔?
            plot_visual_matrix_paper_style(vis_data, args.save_dir, sample_id=i)

    # Save CSV
    csv_data = []
    snr_keys = [k for k in results.keys() if k is not None]
    snr_keys.sort(reverse=True)
    ordered_keys = [None] + snr_keys

    for k in ordered_keys:
        name = 'Clean' if k is None else k
        csv_data.append((name, results[k]))

    df = pd.DataFrame(csv_data, columns=['SNR_dB', 'SSIM'])
    df.to_csv(os.path.join(args.save_dir, 'snr_results_final.csv'), index=False)
    print(f"Done! Check: {args.save_dir}")


if __name__ == '__main__':
    main()
