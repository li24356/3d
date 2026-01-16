import  torch
print(f"PyTorch版本: {torch.__version__}")  # 应包含类似 '2.6.0+cu126' 的标识
print(f"CUDA是否可用: {torch.cuda.is_available()}")  # 应输出 True
print(f"PyTorch使用的CUDA版本: {torch.version.cuda}")  # 应输出 12.6