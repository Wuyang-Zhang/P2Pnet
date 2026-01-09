#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练指标可视化脚本
从run_log.txt文件中解析训练指标并生成可视化图表
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取指标数据

    Args:
        log_file_path: 日志文件路径

    Returns:
        dict: 包含各种指标的字典
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found {log_file_path}")
        return None

    # 使用正则表达式提取所有指标
    epoch_matches = re.findall(r'\[ep (\d+)\]', content)
    loss_matches = re.findall(r'loss/loss@\d+: ([0-9.e-]+)', content)
    mae_matches = re.findall(r'mae:([0-9.e-]+)', content)
    mse_matches = re.findall(r'mse:([0-9.e-]+)', content)
    best_mae_matches = re.findall(r'best mae:([0-9.e-]+)', content)

    # 转换为数值
    epochs = [int(x) for x in epoch_matches]
    train_losses = [float(x) for x in loss_matches]
    mae_values = [float(x) for x in mae_matches]
    mse_values = [float(x) for x in mse_matches]
    best_maes = [float(x) for x in best_mae_matches]

    # 检查数据长度
    min_len = min(len(epochs), len(mae_values), len(mse_values))
    if min_len == 0:
        print("Error: Insufficient metrics data found")
        return None

    # 截取相同长度的数据
    epochs = epochs[:min_len]
    mae_values = mae_values[:min_len]
    mse_values = mse_values[:min_len]

    # 处理训练损失（可能比其他指标少）
    if len(train_losses) < min_len:
        train_losses.extend([0] * (min_len - len(train_losses)))
    else:
        train_losses = train_losses[:min_len]

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'mae_values': mae_values,
        'mse_values': mse_values,
        'best_maes': best_maes if best_maes else [min(mae_values)] * len(mae_values)
    }

def create_plots(data, save_path=None):
    """
    创建指标可视化图表

    Args:
        data: 解析后的数据字典
        save_path: 保存路径，如果为None则显示图表
    """
    epochs = data['epochs']
    train_losses = data['train_losses']
    mae_values = data['mae_values']
    mse_values = data['mse_values']
    best_maes = data['best_maes']

    # 过滤掉值为0的数据点（表示缺失数据）
    valid_indices = [i for i in range(len(mae_values)) if mae_values[i] > 0]

    if not valid_indices:
        print("Error: No valid metrics data found")
        return

    epochs = [epochs[i] for i in valid_indices]
    train_losses = [train_losses[i] for i in valid_indices]
    mae_values = [mae_values[i] for i in valid_indices]
    mse_values = [mse_values[i] for i in valid_indices]

    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('P2PNet Training Metrics Visualization', fontsize=16, fontweight='bold')

    # 1. MAE曲线
    ax1.plot(epochs, mae_values, 'b-', linewidth=2, label='MAE', marker='o', markersize=3)
    if best_maes:
        ax1.axhline(y=min(best_maes), color='r', linestyle='--', linewidth=2,
                   label=f'最佳MAE: {min(best_maes):.3f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error (MAE) Trend')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. MSE曲线
    ax2.plot(epochs, mse_values, 'g-', linewidth=2, label='MSE', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error (MSE) Trend')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 训练损失曲线
    valid_loss_indices = [i for i in range(len(train_losses)) if train_losses[i] > 0]
    if valid_loss_indices:
        loss_epochs = [epochs[i] for i in valid_loss_indices]
        loss_values = [train_losses[i] for i in valid_loss_indices]
        ax3.plot(loss_epochs, loss_values, 'r-', linewidth=2, label='Training Loss',
                marker='^', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss Trend')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. MAE和MSE对比
    ax4.plot(epochs, mae_values, 'b-', linewidth=2, label='MAE', marker='o', markersize=3)
    ax4.plot(epochs, mse_values, 'g-', linewidth=2, label='MSE', marker='s', markersize=3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Error')
    ax4.set_title('MAE vs MSE Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()

def print_summary_stats(data):
    """
    打印训练指标的统计摘要
    """
    mae_values = [x for x in data['mae_values'] if x > 0]
    mse_values = [x for x in data['mse_values'] if x > 0]
    train_losses = [x for x in data['train_losses'] if x > 0]

    if not mae_values:
        print("没有找到有效的指标数据")
        return

    print("\n" + "="*50)
    print("Training Metrics Statistical Summary")
    print("="*50)
    print(f"MAE - Min: {min(mae_values):.3f}, Max: {max(mae_values):.3f}, Final: {mae_values[-1]:.3f}")
    print(f"MSE - Min: {min(mse_values):.3f}, Max: {max(mse_values):.3f}, Final: {mse_values[-1]:.3f}")
    print(f"MAE Improvement: {mae_values[0] - min(mae_values):.3f} ({((mae_values[0] - min(mae_values)) / mae_values[0] * 100):.1f}%)")
    print(f"MSE Improvement: {mse_values[0] - min(mse_values):.3f} ({((mse_values[0] - min(mse_values)) / mse_values[0] * 100):.1f}%)")

    if train_losses:
        print(f"Training Loss - Min: {min(train_losses):.6f}, Max: {max(train_losses):.6f}, Final: {train_losses[-1]:.6f}")

    print(f"Training Epochs: {len(mae_values)}")
    print("="*50)

def main():
    """
    主函数
    """
    log_file = Path("log/run_log.txt")

    if not log_file.exists():
        print(f"错误：找不到日志文件 {log_file}")
        return

    print(f"Parsing log file: {log_file}")

    # 解析日志文件
    data = parse_training_log(log_file)
    if data is None:
        return

    # 打印统计摘要
    print_summary_stats(data)

    # 创建可视化图表
    output_file = "log/training_metrics_visualization.png"
    create_plots(data, save_path=output_file)

if __name__ == "__main__":
    main()