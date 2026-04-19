# 更新日志 / CHANGELOG

## [v1.0.0] - 2026-04-19

### 🎉 首次发布

#### 功能
- ✅ 点击图片进行 SAM 分割
- ✅ 框选区域进行精确分割
- ✅ 自动检测：SAM + ResNet 识别所有区域
- ✅ 自动分割：SAM + ResNet 识别并分割所有物体
- ✅ 图像识别：ResNet50 识别 Top-5 物体
- ✅ 场景分析：亮度、色调、场景类型
- ✅ 点击物体名称查看精确分割掩码

#### 技术栈
- **后端**: FastAPI + SAM (vit_b) + ResNet50
- **前端**: React + OpenSeadragon
- **AI**: SAM (Segment Anything) + ResNet50 (ImageNet)
- **设备**: CUDA (RTX 3060 Laptop)

#### 已知问题
- ResNet 对小区域识别准确率较低
- 分割掩码有时不够精确（SAM 限制）

---

## 版本回退

```bash
# 查看所有版本
git log --oneline

# 回退到指定版本
git checkout <commit_hash>

# 回退并恢复工作区
git checkout -- .
```
