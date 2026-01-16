# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# import time
# import os
# import torch
# import numpy as np
# from pathlib import Path
# import random
#
# # ==========================================
# # 0. 环境与全局设置
# # ==========================================
# # 解决某些环境中 OpenMP 库冲突导致的报错（如 "OMP: Error #15: Initializing libiomp5md.dll"）
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#
# # ==========================================
# # 1. 基础组件模块 (SEBlock, ResBlock)
# # ==========================================
#
# class SEBlock(nn.Module):
#     """
#     Squeeze-and-Excitation (SE) 模块
#     作用：让网络自动学习哪些特征通道（Channel）是重要的，哪些是噪音。
#     原理：
#     1. Squeeze: 通过全局平均池化压缩特征图。
#     2. Excitation: 通过两个全连接层生成权重。
#     3. Scale: 将权重乘回原特征图，增强有效特征，抑制无效特征。
#     """
#
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         # fc1 将通道数压缩（降维），fc2 将通道数恢复（升维）
#         self.fc1 = nn.Linear(channel, channel // reduction)
#         self.fc2 = nn.Linear(channel // reduction, channel)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         batch_size, channel, _, _ = x.size()
#         # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
#         squeezed = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channel)
#
#         # 激励机制：计算通道权重
#         fc1_out = F.relu(self.fc1(squeezed))
#         fc2_out = self.fc2(fc1_out)
#
#         # 生成 0~1 之间的权重系数，并调整形状为 [B, C, 1, 1] 以便进行广播乘法
#         scale = self.sigmoid(fc2_out).view(batch_size, channel, 1, 1)
#
#         # 将权重应用到原始输入上
#         return x * scale.expand_as(x)
#
#
# class ResBlock(nn.Module):
#     """
#     改进的残差块 (Residual Block)
#
#     关键改进：
#     使用 padding_mode='reflect' (镜像填充)。
#
#     原因：
#     普通的 'zeros' (补零) 填充会在图像边缘产生强烈的人工突变（从有信号突然变为0）。
#     卷积核容易将这种突变误认为是“断层”特征，导致边缘误检。
#     镜像填充会复制边缘的像素，过渡更自然，能有效消除这种伪影。
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super(ResBlock, self).__init__()
#
#         # 第一层卷积：注意 padding_mode='reflect'
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
#                                padding_mode='reflect', bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 第二层卷积
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
#                                padding_mode='reflect', bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # Shortcut (捷径连接) 处理：
#         # 如果输入和输出通道数不一致，需要用 1x1 卷积调整维度，以便能够相加
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         residual = self.shortcut(x)  # 备份输入
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual  # 核心：将输入直接加到输出上（残差连接）
#         out = self.relu(out)
#         return out
#
#
# # ==========================================
# # 2. 核心网络架构 (DeepSEResUNet)
# # ==========================================
#
# class DeepSEResUNet(nn.Module):
#     """
#     加深版 U-Net 架构
#
#     结构特点：
#     1. 深度增加：由原来的 4 层编码器增加到 5 层 (包含 Bottleneck)。
#     2. 宽度增加：起始通道数从 16 增加到 32，最深处达到 512。
#     3. 目的：更深的网络拥有更大的“感受野”，能看清地层的大范围走向，
#        从而更好地区分“真正的断层”和“仅仅是地层边缘的纹理”。
#     """
#
#     def __init__(self):
#         super(DeepSEResUNet, self).__init__()
#
#         # --- Encoder (编码器/下采样路径) ---
#         # Level 1: 输入(1) -> 32
#         self.enc1 = ResBlock(1, 32)
#         self.se1 = SEBlock(32)
#         self.pool1 = nn.MaxPool2d(2)  # 尺寸减半
#
#         # Level 2: 32 -> 64
#         self.enc2 = ResBlock(32, 64)
#         self.se2 = SEBlock(64)
#         self.pool2 = nn.MaxPool2d(2)
#
#         # Level 3: 64 -> 128
#         self.enc3 = ResBlock(64, 128)
#         self.se3 = SEBlock(128)
#         self.pool3 = nn.MaxPool2d(2)
#
#         # Level 4: 128 -> 256
#         self.enc4 = ResBlock(128, 256)
#         self.se4 = SEBlock(256)
#         self.pool4 = nn.MaxPool2d(2)
#
#         # --- Bottleneck (瓶颈层/最深层) ---
#         # Level 5: 256 -> 512
#         # 这里不进行池化，是信息的最高级抽象
#         self.bottleneck = ResBlock(256, 512)
#         self.se_bot = SEBlock(512)
#
#         # --- Decoder (解码器/上采样路径) ---
#         # Up 4: 512 -> 256
#         self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 反卷积放大尺寸
#         self.dec4 = ResBlock(512,
#                              256)  # 输入是 concat(256+256)=512, 输出 256 (这里ResBlock内部会自动处理in!=out的情况或者你需要修改ResBlock输入为512)
#         # 修正：ResBlock的输入维度应该是 concat 之后的维度。
#         # 此处 ResBlock(512, 256) 是指输入512通道(来自跳跃连接+上采样)，压缩回256
#         self.se_dec4 = SEBlock(256)
#
#         # Up 3: 256 -> 128
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = ResBlock(256, 128)  # concat(128+128)=256 -> 128
#         self.se_dec3 = SEBlock(128)
#
#         # Up 2: 128 -> 64
#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = ResBlock(128, 64)  # concat(64+64)=128 -> 64
#         self.se_dec2 = SEBlock(64)
#
#         # Up 1: 64 -> 32
#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = ResBlock(64, 32)  # concat(32+32)=64 -> 32
#         self.se_dec1 = SEBlock(32)
#
#         # Output Layer: 输出单通道 (Probability Map)
#         self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
#
#     def forward(self, x):
#         # --- 编码过程 ---
#         x1 = self.se1(self.enc1(x))
#         x1_p = self.pool1(x1)
#
#         x2 = self.se2(self.enc2(x1_p))
#         x2_p = self.pool2(x2)
#
#         x3 = self.se3(self.enc3(x2_p))
#         x3_p = self.pool3(x3)
#
#         x4 = self.se4(self.enc4(x3_p))
#         x4_p = self.pool4(x4)
#
#         # --- 瓶颈层 ---
#         x_bot = self.se_bot(self.bottleneck(x4_p))
#
#         # --- 解码过程 (含跳跃连接 Skip Connections) ---
#         d4 = self.up4(x_bot)
#         d4 = torch.cat([d4, x4], dim=1)  # 将编码器的特征 x4 拼接到解码器
#         d4 = self.se_dec4(self.dec4(d4))
#
#         d3 = self.up3(d4)
#         d3 = torch.cat([d3, x3], dim=1)
#         d3 = self.se_dec3(self.dec3(d3))
#
#         d2 = self.up2(d3)
#         d2 = torch.cat([d2, x2], dim=1)
#         d2 = self.se_dec2(self.dec2(d2))
#
#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, x1], dim=1)
#         d1 = self.se_dec1(self.dec1(d1))
#
#         return self.final_conv(d1)
#
#
# # ==========================================
# # 3. 损失函数 (Focal + Dice Loss)
# # ==========================================
#
# class FocalDiceLoss(nn.Module):
#     """
#     混合损失函数，专门解决断层检测中的“类别极度不平衡”问题。
#     1. Focal Loss: 关注“难分类”的样本，降低简单背景样本的权重。
#     2. Dice Loss: 关注预测区域和真实区域的重叠度（IOU），对细长形状的断层非常有效。
#     """
#
#     def __init__(self, alpha=0.5, gamma=2, smooth=1.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 输出未缩减的Loss以便加权
#
#     def forward(self, inputs, targets):
#         # --- 计算 Focal Loss ---
#         bce_loss = self.bce(inputs, targets)
#         probas = torch.sigmoid(inputs)
#         loss_pt = torch.exp(-bce_loss)  # p_t
#         # Focal Loss 公式: -alpha * (1-pt)^gamma * log(pt)
#         focal_loss = (self.alpha * (1 - loss_pt) ** self.gamma * bce_loss).mean()
#
#         # --- 计算 Dice Loss ---
#         # 将 Tensor 拉平成一维向量计算交集
#         intersection = (probas.view(-1) * targets.view(-1)).sum()
#         # Dice 系数公式: (2 * 交集 + smooth) / (预测面积 + 真实面积 + smooth)
#         dice_loss = 1 - (2. * intersection + self.smooth) / (probas.sum() + targets.sum() + self.smooth)
#
#         return focal_loss + dice_loss
#
#
# # ==========================================
# # 4. 数据集加载与增强 (MyDataset)
# # ==========================================
#
# class MyDataset(Dataset):
#     """
#     自定义数据集类，负责读取 .dat 文件并应用增强。
#     """
#
#     def __init__(self, data_folder, label_folder, augment=True, noise=False):
#         # 搜索所有 .dat 文件并按数字顺序排序
#         self.data_files = sorted(Path(data_folder).rglob('*.dat'), key=lambda x: int(x.stem))
#         self.label_files = sorted(Path(label_folder).rglob('*.dat'), key=lambda x: int(x.stem))
#         self.augment = augment  # 是否开启增强
#         self.noise = noise  # 是否开启噪声
#         # 基础增强：每个样本都会被视为 4 个样本（0, 90, 180, 270度旋转）
#         self.rotation_angles = [0, 90, 180, 270] if augment else [0]
#
#     def __len__(self):
#         # 数据集长度 = 文件数 * 旋转角度数
#         return len(self.data_files) * len(self.rotation_angles)
#
#     def __getitem__(self, idx):
#         # 计算当前索引对应的文件索引和旋转角度
#         file_idx = idx // len(self.rotation_angles)
#         angle = self.rotation_angles[idx % len(self.rotation_angles)]
#
#         # 读取二进制数据并 reshape 为 [1, 128, 128]
#         data = np.fromfile(self.data_files[file_idx], dtype=np.float32).reshape(1, 128, 128)
#         label = np.fromfile(self.label_files[file_idx], dtype=np.float32).reshape(1, 128, 128)
#
#         # 1. 基础旋转 (Rotation)
#         if angle != 0:
#             k = angle // 90
#             data = np.rot90(data, k=k, axes=(1, 2)).copy()
#             label = np.rot90(label, k=k, axes=(1, 2)).copy()
#
#         # 2. 随机增强 (Augmentation)
#         if self.augment:
#             # 随机翻转 (Flip) - 水平或垂直
#             if random.random() > 0.5:
#                 data = np.flip(data, axis=2).copy();
#                 label = np.flip(label, axis=2).copy()
#             if random.random() > 0.5:
#                 data = np.flip(data, axis=1).copy();
#                 label = np.flip(label, axis=1).copy()
#
#             # 随机增益 (Gain) - 模拟地震振幅强弱
#             data = data * np.random.uniform(0.8, 1.2)
#
#             # **Cutout (随机遮挡)** - 核心增强
#             # 逻辑：随机把 Input 挖掉一块，但 Label 保持不变。
#             # 作用：强迫模型根据上下文推断断层的走向，解决断层预测“断断续续”的问题。
#             if random.random() > 0.4:  # 60% 概率触发
#                 ms = random.randint(15, 35)  # 遮挡块大小
#                 y, x = random.randint(0, 128 - ms), random.randint(0, 128 - ms)
#                 data[:, y:y + ms, x:x + ms] = 0.0  # 只遮挡 data，不遮挡 label
#
#         # 3. 随机噪声 (Noise)
#         if self.noise:
#             data += np.random.normal(0, 0.4, data.shape)
#
#         # 4. 标准化 (Z-Score Normalization)
#         # 这一步非常重要，确保数据分布在 0 附近，加速网络收敛
#         data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
#
#         return torch.from_numpy(data_norm).float(), torch.from_numpy(label).float(), self.data_files[file_idx].name
#
#
# def calculate_accuracy(output, label):
#     """ 计算二分类准确率 (阈值 0.5) """
#     predicted = (torch.sigmoid(output) > 0.5).float()
#     return (predicted == label).sum().item() / label.numel()
#
#
# # ==========================================
# # 5. 主程序 (训练、验证、可视化)
# # ==========================================
#
# if __name__ == '__main__':
#     # --- 超参数设置 ---
#     BATCH_SIZE = 12
#     INITIAL_LR = 1e-3
#     EPOCHS = 100
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     MODEL_PATH = 'DeepSEResUNet_best.pth'  # 保存模型的文件名
#
#     # --- 数据准备 ---
#     # 实例化数据集，开启增强和噪声
#     full_dataset = MyDataset('data', 'label', augment=True, noise=True)
#
#     # 划分训练集和验证集 (80% 训练, 20% 验证)
#     train_size = int(len(full_dataset) * 0.9)
#     train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
#
#     # 定义 DataLoader
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
#
#     # --- 模型初始化 ---
#     model = DeepSEResUNet().to(DEVICE)
#
#     # 使用 AdamW 优化器 (比 Adam 泛化性更好)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-4)
#
#     # --- 学习率调度器 (ReduceLROnPlateau) ---
#     # 监控验证集 Loss，如果 10 个 epoch 不下降，就将学习率乘以 0.9
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.8, patience=10
#     )
#
#     criterion = FocalDiceLoss()
#     best_val_loss = float('inf')
#
#     # --- 训练循环 ---
#     # 如果本地没有模型文件，则开始训练
#     if not os.path.exists(MODEL_PATH):
#         print("开始训练 Deep-SE-ResUNet...")
#         # 计算并打印模型参数量
#         print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
#
#         for epoch in range(EPOCHS):
#             # 1. 训练阶段
#             model.train()
#             train_l = 0
#             for d, l, _ in train_loader:
#                 d, l = d.to(DEVICE), l.to(DEVICE)  # 搬运数据到 GPU
#                 optimizer.zero_grad()  # 梯度清零
#                 out = model(d)  # 前向传播
#                 loss = criterion(out, l)  # 计算损失
#                 loss.backward()  # 反向传播
#                 optimizer.step()  # 更新权重
#                 train_l += loss.item() * d.size(0)  # 累加 Loss
#
#             # 2. 验证阶段
#             model.eval()
#             val_l, val_acc = 0, 0
#             with torch.no_grad():  # 验证时不计算梯度，节省显存
#                 for d, l, _ in val_loader:
#                     d, l = d.to(DEVICE), l.to(DEVICE)
#                     out = model(d)
#                     val_l += criterion(out, l).item() * d.size(0)
#                     val_acc += calculate_accuracy(out, l) * d.size(0)
#
#             # 3. 计算平均指标
#             avg_train_loss = train_l / len(train_ds)
#             avg_val_loss = val_l / len(val_ds)
#
#             # 4. 更新学习率
#             # 调度器会根据当前的 avg_val_loss 决定是否降低学习率
#             scheduler.step(avg_val_loss)
#
#             # 获取当前实际学习率
#             current_lr = optimizer.param_groups[0]['lr']
#             print(
#                 f"Epoch {epoch + 1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
#
#             # 5. 保存最佳模型
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 torch.save(model.state_dict(), MODEL_PATH)
#                 print(">>> 已保存最佳模型")
#     else:
#         print(f"检测到模型文件 {MODEL_PATH}，直接加载进行预测...")
#
#     # --- 结果可视化 ---
#     model.load_state_dict(torch.load(MODEL_PATH))  # 加载最佳权重
#     model.eval()
#
#     print("开始可视化前3张验证集结果...")
#     with torch.no_grad():
#         for i, (data, label, name) in enumerate(val_loader):
#             if i >= 3: break  # 只看前3张
#
#             out = model(data.to(DEVICE))
#             # 将 Logits 转为概率，再转为 0/1 掩码
#             pred = (torch.sigmoid(out) > 0.5).float().cpu().squeeze().numpy()
#
#             # 绘图
#             fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#             ax[0].imshow(data.squeeze().numpy().T, cmap='gray');
#             ax[0].set_title("Input (输入地震数据)")
#             ax[1].imshow(label.squeeze().numpy().T, cmap='gray');
#             ax[1].set_title("Target (真实断层标签)")
#             ax[2].imshow(pred.T, cmap='gray');
#             ax[2].set_title("Prediction (模型预测)")
#             plt.show()
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# =========================================================
# 0. 环境 & 随机种子
# =========================================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================
# 1. 基础模块（与你原来一致）
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1,
                               padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                               padding_mode='reflect', bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


# =========================================================
# 2. 网络（完全保持你原结构）
# =========================================================
class DeepSEResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResBlock(1, 32);  self.se1 = SEBlock(32)
        self.enc2 = ResBlock(32, 64); self.se2 = SEBlock(64)
        self.enc3 = ResBlock(64, 128);self.se3 = SEBlock(128)
        self.enc4 = ResBlock(128,256);self.se4 = SEBlock(256)

        self.pool = nn.MaxPool2d(2)

        self.bot = ResBlock(256,512); self.seb = SEBlock(512)

        self.up4 = nn.ConvTranspose2d(512,256,2,2)
        self.dec4 = ResBlock(512,256); self.sed4 = SEBlock(256)

        self.up3 = nn.ConvTranspose2d(256,128,2,2)
        self.dec3 = ResBlock(256,128); self.sed3 = SEBlock(128)

        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = ResBlock(128,64);  self.sed2 = SEBlock(64)

        self.up1 = nn.ConvTranspose2d(64,32,2,2)
        self.dec1 = ResBlock(64,32);   self.sed1 = SEBlock(32)

        self.outc = nn.Conv2d(32,1,1)

    def forward(self, x):
        x1 = self.se1(self.enc1(x))
        x2 = self.se2(self.enc2(self.pool(x1)))
        x3 = self.se3(self.enc3(self.pool(x2)))
        x4 = self.se4(self.enc4(self.pool(x3)))

        xb = self.seb(self.bot(self.pool(x4)))

        d4 = self.sed4(self.dec4(torch.cat([self.up4(xb), x4],1)))
        d3 = self.sed3(self.dec3(torch.cat([self.up3(d4), x3],1)))
        d2 = self.sed2(self.dec2(torch.cat([self.up2(d3), x2],1)))
        d1 = self.sed1(self.dec1(torch.cat([self.up1(d2), x1],1)))

        return self.outc(d1)


# =========================================================
# 3. Loss
# =========================================================
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal = focal.mean()

        prob = torch.sigmoid(logits)
        inter = (prob * targets).sum()
        dice = 1 - (2*inter + self.smooth) / \
               (prob.sum() + targets.sum() + self.smooth)

        return focal + dice


# =========================================================
# 4. Dataset（★ 关键：train / val 分离）
# =========================================================
class SeismicDataset(Dataset):
    def __init__(self, data_files, label_files,
                 augment=False, noise=False):
        self.data_files = data_files
        self.label_files = label_files
        self.augment = augment
        self.noise = noise
        self.rotations = [0,90,180,270] if augment else [0]

    def __len__(self):
        return len(self.data_files) * len(self.rotations)

    def __getitem__(self, idx):
        fid = idx // len(self.rotations)
        angle = self.rotations[idx % len(self.rotations)]

        data = np.fromfile(self.data_files[fid], np.float32).reshape(1,128,128)
        label = np.fromfile(self.label_files[fid], np.float32).reshape(1,128,128)

        if angle:
            k = angle // 90
            data = np.rot90(data, k, (1,2)).copy()
            label = np.rot90(label, k, (1,2)).copy()

        if self.augment:
            if random.random()>0.5:
                data = np.flip(data,2).copy()
                label = np.flip(label,2).copy()
            if random.random()>0.5:
                data = np.flip(data,1).copy()
                label = np.flip(label,1).copy()
            data *= np.random.uniform(0.8,1.2)
            if random.random()>0.4:
                ms = random.randint(15,35)
                y,x = random.randint(0,128-ms), random.randint(0,128-ms)
                data[:,y:y+ms,x:x+ms]=0

        if self.noise:
            data += np.random.normal(0,0.4,data.shape)

        data = (data - data.mean())/(data.std()+1e-8)
        return torch.tensor(data), torch.tensor(label)


# =========================================================
# 5. Metrics（Dice / IoU / Precision / Recall / F1）
# =========================================================
def seg_metrics(logits, targets, thr=0.5, eps=1e-8):
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * targets).sum()
    fp = (pred * (1-targets)).sum()
    fn = ((1-pred) * targets).sum()

    dice = (2*tp+eps)/(2*tp+fp+fn+eps)
    iou  = (tp+eps)/(tp+fp+fn+eps)
    prec = (tp+eps)/(tp+fp+eps)
    rec  = (tp+eps)/(tp+fn+eps)
    f1   = 2*prec*rec/(prec+rec+eps)
    return dice, iou, prec, rec, f1


# =========================================================
# 6. 主训练（AMP）
# =========================================================
def main():
    EPOCHS = 100
    BATCH = 5
    LR = 1e-3
    MODEL_PATH = "DeepSEResUNet_best_4.pth"

    all_data = sorted(Path('data').glob('*.dat'))
    all_label = sorted(Path('label').glob('*.dat'))

    idx = list(range(len(all_data)))
    random.shuffle(idx)
    split = int(0.95*len(idx))

    train_id, val_id = idx[:split], idx[split:]

    train_ds = SeismicDataset(
        [all_data[i] for i in train_id],
        [all_label[i] for i in train_id],
        augment=True, noise=True
    )
    val_ds = SeismicDataset(
        [all_data[i] for i in val_id],
        [all_label[i] for i in val_id],
        augment=False, noise=False
    )

    train_loader = DataLoader(train_ds, BATCH, True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, 1, False)

    model = DeepSEResUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        patience=4,
        factor=0.6
    )
    loss_fn = FocalDiceLoss()
    scaler = GradScaler()

    best = 1e9
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with autocast():
                out = model(x)
                loss = loss_fn(out,y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()*x.size(0)

        model.eval()
        vl=0; md=mi=mp=mr=mf=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(DEVICE),y.to(DEVICE)
                out=model(x)
                vl+=loss_fn(out,y).item()
                d,i,p,r,f=seg_metrics(out,y)
                md+=d; mi+=i; mp+=p; mr+=r; mf+=f

        sch.step(vl)
        cur_lr = opt.param_groups[0]['lr']
        print(f"E{ep+1:03d} "
              f"LR {cur_lr:.5f} "
              f"Train {tl/len(train_ds):.4f} "
              f"Val {vl/len(val_ds):.4f} "
              f"Dice {md/len(val_ds):.3f} "
              f"IoU {mi/len(val_ds):.3f} "
              f"F1 {mf/len(val_ds):.3f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), MODEL_PATH)
            print(">>> Save best model")

if __name__ == "__main__":
    main()
