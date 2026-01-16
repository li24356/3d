import time
import traceback
import torch
from pathlib import Path

# 从 train.py 导入构造器与配置，确保 train.py 在同一工作区
from train import build_model_from_config, MODEL_CONFIG

# 测试设置（按需修改）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patch_size = (128, 128, 128)   # 测试输入体素尺寸 (D,H,W) — 与训练时一致
test_batch_list = [1, 2, 4]    # 要测试的 batch_size 列表（按顺序测试）
repeats = 3                    # 每个 batch 测试重复次数取平均

# 修改：更可靠的 try_batch（包含 warm-up 与 CUDA 同步），并把 repeats 默认改为 3
def try_batch(model, B, device, patch, warmup=2):
    C = MODEL_CONFIG.get('in_channels', 1)
    D, H, W = patch
    try:
        model.eval()
        x = torch.randn(B, C, D, H, W, device=device)

        # CUDA 下清理并重置峰值统计
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # warm-up 以触发 cuDNN autotune、分配等
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            # 同步并测时
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                t0 = time.time()
                _ = model(x)
                torch.cuda.synchronize(device)
                t1 = time.time()
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            else:
                t0 = time.time()
                _ = model(x)
                t1 = time.time()
                peak_gb = None

        return True, (t1 - t0), peak_gb
    except RuntimeError as e:
        # 捕获 OOM 等运行时错误并清理
        if device.type == 'cuda' and 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
        return False, None, None

def main():
    print("Device:", device)
    print("MODEL_CONFIG:", MODEL_CONFIG.get('model_name', 'unknown'), "base_channels:", MODEL_CONFIG.get('base_channels'))
    # 构建模型（与训练一致）
    model = build_model_from_config(MODEL_CONFIG)
    model = model.to(device)

    success_results = {}
    for B in test_batch_list:
        ok = True
        times = []
        peaks = []
        for r in range(repeats):
            ok_single, t, peak = try_batch(model, B, device, patch_size)
            if not ok_single:
                ok = False
                break
            times.append(t)
            peaks.append(peak if peak is not None else 0.0)
        if ok:
            avg_t = sum(times) / len(times) if times else None
            avg_peak = sum(peaks) / len(peaks) if peaks else None
            success_results[B] = (avg_t, avg_peak)
            print(f"batch={B} OK, time={avg_t:.3f}s, peak_gpu={avg_peak:.3f} GB")
        else:
            print(f"batch={B} FAILED (likely OOM)")
            break

    if len(success_results):
        max_ok = max(success_results.keys())
        print(f"\n推荐可用最大 batch_size = {max_ok}")
        if device.type == 'cuda':
            print("峰值显存参考：")
            for b, (t, p) in success_results.items():
                print(f"  batch={b}: peak ~ {p:.2f} GB, time ~ {t:.3f}s")
    else:
        print("\n所有测试的 batch_size 都失败。建议：")
        print(" - 把 MODEL_CONFIG['base_channels'] 减小（例如 16 -> 8）")
        print(" - 保持 batch_size=1 并使用梯度累积(accum_steps>1)")
        print(" - 开启 use_amp=True 或使用更小模型")

if __name__ == '__main__':
    main()