"""
可视化合成地震数据和断层标签

支持：
- 多方向切片对比
- 3D 可视化
- 叠加显示
- 统计分析
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_sample(seis_path, fault_path):
    """加载地震数据和断层标签"""
    seis = np.load(seis_path)
    fault = np.load(fault_path)
    return seis, fault

def plot_comprehensive_view(seis, fault, sample_idx, save_path=None):
    """
    绘制综合视图：原始数据 + 断层标签 + 叠加
    """
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)
    
    z_mid = seis.shape[0] // 2
    y_mid = seis.shape[1] // 2
    x_mid = seis.shape[2] // 2
    
    # 标题
    fig.suptitle(f'样本 {sample_idx} - 地震数据和断层标签综合视图', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Z方向切片（水平层）
    row = 0
    # 地震数据
    ax = fig.add_subplot(gs[row, 0])
    im = ax.imshow(seis[z_mid, :, :], cmap='seismic', aspect='auto')
    ax.set_title(f'地震数据 - Z={z_mid}层\n(水平切片)', fontsize=11)
    ax.set_ylabel('Y (crossline)')
    ax.set_xlabel('X (inline)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 断层标签
    ax = fig.add_subplot(gs[row, 1])
    im = ax.imshow(fault[z_mid, :, :], cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'断层标签 - Z={z_mid}层\n(0=非断层, 1=断层)', fontsize=11)
    ax.set_ylabel('Y (crossline)')
    ax.set_xlabel('X (inline)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    
    # 叠加显示
    ax = fig.add_subplot(gs[row, 2])
    ax.imshow(seis[z_mid, :, :], cmap='gray', aspect='auto')
    # 将断层用半透明红色叠加
    fault_overlay = np.ma.masked_where(fault[z_mid, :, :] == 0, fault[z_mid, :, :])
    im = ax.imshow(fault_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'叠加显示 - Z={z_mid}层\n(红色=断层)', fontsize=11)
    ax.set_ylabel('Y (crossline)')
    ax.set_xlabel('X (inline)')
    
    # Y方向切片（纵断面，沿crossline方向）
    row = 1
    # 地震数据
    ax = fig.add_subplot(gs[row, 0])
    im = ax.imshow(seis[:, y_mid, :], cmap='seismic', aspect='auto')
    ax.set_title(f'地震数据 - Y={y_mid}剖面\n(纵断面)', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('X (inline)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 断层标签
    ax = fig.add_subplot(gs[row, 1])
    im = ax.imshow(fault[:, y_mid, :], cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'断层标签 - Y={y_mid}剖面', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('X (inline)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    
    # 叠加显示
    ax = fig.add_subplot(gs[row, 2])
    ax.imshow(seis[:, y_mid, :], cmap='gray', aspect='auto')
    fault_overlay = np.ma.masked_where(fault[:, y_mid, :] == 0, fault[:, y_mid, :])
    im = ax.imshow(fault_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'叠加显示 - Y={y_mid}剖面', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('X (inline)')
    ax.invert_yaxis()
    
    # X方向切片（纵断面，沿inline方向）
    row = 2
    # 地震数据
    ax = fig.add_subplot(gs[row, 0])
    im = ax.imshow(seis[:, :, x_mid], cmap='seismic', aspect='auto')
    ax.set_title(f'地震数据 - X={x_mid}剖面\n(纵断面)', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('Y (crossline)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 断层标签
    ax = fig.add_subplot(gs[row, 1])
    im = ax.imshow(fault[:, :, x_mid], cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'断层标签 - X={x_mid}剖面', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('Y (crossline)')
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    
    # 叠加显示
    ax = fig.add_subplot(gs[row, 2])
    ax.imshow(seis[:, :, x_mid], cmap='gray', aspect='auto')
    fault_overlay = np.ma.masked_where(fault[:, :, x_mid] == 0, fault[:, :, x_mid])
    im = ax.imshow(fault_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'叠加显示 - X={x_mid}剖面', fontsize=11)
    ax.set_ylabel('Z (深度)')
    ax.set_xlabel('Y (crossline)')
    ax.invert_yaxis()
    
    # 统计信息
    row = 3
    ax = fig.add_subplot(gs[row, :])
    ax.axis('off')
    
    # 计算统计信息
    fault_ratio = fault.mean() * 100
    fault_count = fault.sum()
    total_voxels = fault.size
    
    stats_text = f"""
    数据统计信息:
    
    地震数据:
      • 形状: {seis.shape} (Z × Y × X) = {seis.size:,} 体素
      • 数据范围: [{seis.min():.3f}, {seis.max():.3f}]
      • 均值: {seis.mean():.3f}, 标准差: {seis.std():.3f}
    
    断层标签:
      • 形状: {fault.shape} (Z × Y × X) = {fault.size:,} 体素
      • 断层像素数: {fault_count:,} ({fault_ratio:.2f}%)
      • 非断层像素数: {total_voxels - fault_count:,} ({100-fault_ratio:.2f}%)
      • 断层/非断层比例: 1:{(total_voxels-fault_count)/max(fault_count, 1):.1f}
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存到: {save_path}")
    
    return fig

def plot_multi_depth_slices(seis, fault, sample_idx, num_slices=5, save_path=None):
    """
    绘制多个深度层的切片
    """
    fig, axes = plt.subplots(2, num_slices, figsize=(4*num_slices, 8))
    fig.suptitle(f'样本 {sample_idx} - 多深度层切片对比', fontsize=14, fontweight='bold')
    
    z_indices = np.linspace(seis.shape[0]//4, 3*seis.shape[0]//4, num_slices, dtype=int)
    
    for i, z_idx in enumerate(z_indices):
        # 地震数据
        axes[0, i].imshow(seis[z_idx, :, :], cmap='seismic', aspect='auto')
        axes[0, i].set_title(f'地震 Z={z_idx}')
        axes[0, i].axis('off')
        
        # 叠加显示
        axes[1, i].imshow(seis[z_idx, :, :], cmap='gray', aspect='auto')
        fault_overlay = np.ma.masked_where(fault[z_idx, :, :] == 0, fault[z_idx, :, :])
        axes[1, i].imshow(fault_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1, aspect='auto')
        axes[1, i].set_title(f'断层标注 Z={z_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存到: {save_path}")
    
    return fig

def plot_histogram_analysis(seis, fault, sample_idx, save_path=None):
    """
    绘制数据分布直方图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'样本 {sample_idx} - 数据分布分析', fontsize=14, fontweight='bold')
    
    # 地震数据直方图
    axes[0, 0].hist(seis.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('地震数据振幅分布')
    axes[0, 0].set_xlabel('振幅值')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(seis.mean(), color='r', linestyle='--', label=f'均值={seis.mean():.3f}')
    axes[0, 0].legend()
    
    # 断层标签分布
    fault_values, fault_counts = np.unique(fault, return_counts=True)
    axes[0, 1].bar(fault_values, fault_counts, color=['skyblue', 'coral'], edgecolor='black')
    axes[0, 1].set_title('断层标签分布')
    axes[0, 1].set_xlabel('类别 (0=非断层, 1=断层)')
    axes[0, 1].set_ylabel('体素数量')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 为每个柱子添加数值标签
    for i, (val, count) in enumerate(zip(fault_values, fault_counts)):
        axes[0, 1].text(val, count, f'{count:,}\n({count/fault.size*100:.1f}%)', 
                       ha='center', va='bottom')
    
    # 断层与非断层区域的地震数据分布对比
    seis_fault = seis[fault == 1]
    seis_non_fault = seis[fault == 0]
    
    axes[1, 0].hist(seis_non_fault.flatten(), bins=50, alpha=0.6, 
                    color='blue', label='非断层区域', edgecolor='black')
    axes[1, 0].hist(seis_fault.flatten(), bins=50, alpha=0.6, 
                    color='red', label='断层区域', edgecolor='black')
    axes[1, 0].set_title('断层与非断层区域的地震振幅对比')
    axes[1, 0].set_xlabel('振幅值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 深度方向的断层分布
    fault_per_depth = fault.sum(axis=(1, 2))  # 每个深度层的断层数量
    axes[1, 1].plot(fault_per_depth, range(len(fault_per_depth)), linewidth=2, color='darkred')
    axes[1, 1].fill_betweenx(range(len(fault_per_depth)), 0, fault_per_depth, alpha=0.3, color='red')
    axes[1, 1].set_title('深度方向断层分布')
    axes[1, 1].set_xlabel('断层像素数')
    axes[1, 1].set_ylabel('深度 (Z)')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存到: {save_path}")
    
    return fig

def visualize_all_samples(data_dir='demo_output', output_dir='visualizations'):
    """
    可视化所有样本
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("地震数据可视化")
    print(f"{'='*70}\n")
    
    # 查找所有样本
    seis_files = sorted(data_dir.glob('seis_*.npy'))
    
    if not seis_files:
        print(f"⚠️  在 {data_dir} 中未找到数据文件")
        return
    
    print(f"找到 {len(seis_files)} 个样本\n")
    
    for seis_path in seis_files:
        sample_idx = seis_path.stem.split('_')[1]
        fault_path = data_dir / f'fault_{sample_idx}.npy'
        
        if not fault_path.exists():
            print(f"⚠️  跳过样本 {sample_idx}: 缺少断层标签")
            continue
        
        print(f"处理样本 {sample_idx}...")
        
        # 加载数据
        seis, fault = load_sample(seis_path, fault_path)
        
        # 综合视图
        print(f"  生成综合视图...")
        plot_comprehensive_view(
            seis, fault, sample_idx,
            save_path=output_dir / f'sample_{sample_idx}_comprehensive.png'
        )
        plt.close()
        
        # 多深度切片
        print(f"  生成多深度切片...")
        plot_multi_depth_slices(
            seis, fault, sample_idx,
            save_path=output_dir / f'sample_{sample_idx}_multi_depth.png'
        )
        plt.close()
        
        # 统计分析
        print(f"  生成统计分析图...")
        plot_histogram_analysis(
            seis, fault, sample_idx,
            save_path=output_dir / f'sample_{sample_idx}_statistics.png'
        )
        plt.close()
        
        print(f"  ✓ 完成\n")
    
    print(f"{'='*70}")
    print(f"✓ 所有可视化已保存到: {output_dir}")
    print(f"{'='*70}\n")
    print("生成的文件:")
    print("  • *_comprehensive.png - 综合三维切片视图")
    print("  • *_multi_depth.png - 多深度层对比")
    print("  • *_statistics.png - 数据分布统计")
    print(f"{'='*70}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化地震数据和断层标签')
    parser.add_argument('--data-dir', default='demo_output',
                       help='数据目录（默认：demo_output）')
    parser.add_argument('--output-dir', default='visualizations',
                       help='输出目录（默认：visualizations）')
    parser.add_argument('--sample', type=str, default=None,
                       help='只可视化指定样本（如：000）')
    
    args = parser.parse_args()
    
    if args.sample:
        # 可视化单个样本
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        seis_path = data_dir / f'seis_{args.sample}.npy'
        fault_path = data_dir / f'fault_{args.sample}.npy'
        
        if not seis_path.exists() or not fault_path.exists():
            print(f"✗ 找不到样本 {args.sample}")
            sys.exit(1)
        
        print(f"可视化样本 {args.sample}...")
        seis, fault = load_sample(seis_path, fault_path)
        
        plot_comprehensive_view(seis, fault, args.sample,
                              save_path=output_dir / f'sample_{args.sample}_comprehensive.png')
        plot_multi_depth_slices(seis, fault, args.sample,
                               save_path=output_dir / f'sample_{args.sample}_multi_depth.png')
        plot_histogram_analysis(seis, fault, args.sample,
                               save_path=output_dir / f'sample_{args.sample}_statistics.png')
        
        print(f"✓ 完成，输出保存到: {output_dir}")
    else:
        # 可视化所有样本
        visualize_all_samples(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()
