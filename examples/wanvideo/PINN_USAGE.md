# PINN视频生成 - 快速使用指南

## 🚀 5分钟快速开始

### 1. 运行演示

```bash
cd examples/wanvideo
python pinn_quickstart.py
```

这会生成3个示例视频：
- `pinn_demo_water.mp4` - 瀑布水流 (流体)
- `pinn_demo_ball.mp4` - 球体掉落 (刚体)
- `pinn_demo_cloth.mp4` - 布料摆动 (弹性体)

### 2. 训练你的第一个PINN插件

```bash
cd pinn_training

# 训练流体物理插件 (5-10分钟)
python train_pinn.py \
    --prompt "water splashing" \
    --material_type fluid \
    --num_train_steps 500 \
    --physics_weight 0.1 \
    --output_dir ./outputs_first_try
```

**重要**：原始模型不会被修改！这只是训练一个小插件。

### 3. 使用训练好的插件

```bash
cd pinn_inference

python inference_pinn.py \
    --prompt "water flowing down" \
    --checkpoint_path ../pinn_training/outputs_first_try/pinn_plugin_final.pt \
    --output water_with_physics.mp4
```

## 📊 关键概念

### 什么是PINN插件？

```
原始Wan模型 (14B参数, 冻结)
    ↓
  + PINN插件 (~250K参数, 可训练)
    ↓
  物理约束的视频生成
```

- ✅ 不修改原模型
- ✅ 可随时启用/禁用
- ✅ 插件大小仅为原模型的 0.002%

### 四种材质类型

| 材质 | 关键词 | 物理约束 |
|-----|-------|---------|
| **流体** | water, liquid, flow | 不可压缩、粘性 |
| **刚体** | ball, box, rock | 不变形、动量守恒 |
| **弹性体** | cloth, rubber | 弹性恢复力 |
| **颗粒** | sand, dust | 接触力、摩擦力 |

### 参数调优速查表

| 场景 | physics_weight | warmup_steps | 训练步数 |
|-----|---------------|--------------|---------|
| 简单测试 | 0.05 | 50 | 500 |
| **标准训练** | **0.1** | **100** | **1000** |
| 强约束 | 0.2 | 200 | 2000 |

## 🎯 常见场景示例

### 流体场景

```bash
# 瀑布
python inference_pinn.py --prompt "waterfall flowing down from cliff"

# 液体飞溅
python inference_pinn.py --prompt "water splash in slow motion"

# 烟雾上升
python inference_pinn.py --prompt "smoke rising from chimney"
```

### 刚体场景

```bash
# 球体掉落
python inference_pinn.py --prompt "red ball falling and bouncing"

# 物体碰撞
python inference_pinn.py --prompt "two boxes colliding with each other"

# 多物体堆叠
python inference_pinn.py --prompt "stack of cubes falling down"
```

### 弹性体场景

```bash
# 布料摆动
python inference_pinn.py --prompt "silk cloth waving in wind"

# 橡胶拉伸
python inference_pinn.py --prompt "rubber band being stretched"

# 弹性碰撞
python inference_pinn.py --prompt "elastic ball bouncing repeatedly"
```

## 🔧 故障排除

### 问题1：CUDA内存不足

**解决**：
```bash
# 使用较小分辨率
python inference_pinn.py --height 360 --width 640 --num_frames 41

# 或启用tiled模式（默认已启用）
```

### 问题2：生成质量不如原模型

**原因**：物理权重太大，抑制了生成多样性

**解决**：
```bash
# 降低物理权重
python train_pinn.py --physics_weight 0.05

# 或者推理时禁用物理约束
python inference_pinn.py --prompt "..." 
# （不加 --checkpoint_path，使用原模型）
```

### 问题3：材质识别错误

**解决**：
```bash
# 手动指定材质类型
python train_pinn.py --material_type fluid  # 而不是 auto
```

## 📈 进阶使用

### 对比实验

```bash
# 1. 无物理约束（baseline）
python inference_pinn.py \
    --prompt "water flowing" \
    --output baseline.mp4

# 2. 有物理约束
python inference_pinn.py \
    --prompt "water flowing" \
    --checkpoint_path path/to/plugin.pt \
    --output with_physics.mp4
```

### 混合材质

```bash
# 训练混合材质场景
python train_pinn.py \
    --prompt "water splashing on rocks" \
    --material_type mixed \
    --physics_weight 0.15
```

### 查看训练进度

训练时会自动保存测试视频：
```
outputs_pinn/
├── pinn_plugin_step_100.pt      # 第100步checkpoint
├── test_video_step_100.mp4      # 对应的测试视频
├── pinn_plugin_step_200.pt
├── test_video_step_200.mp4
...
```

对比这些视频观察物理约束的效果。

## 💡 最佳实践

### 1. 从小权重开始

```bash
# 第一次实验
--physics_weight 0.05

# 观察效果后逐步增加
--physics_weight 0.1
--physics_weight 0.15
```

### 2. 使用预热

```bash
# 物理损失逐渐增加，避免训练初期不稳定
--physics_warmup_steps 100
```

### 3. 定期检查测试视频

```bash
# 每100步保存一次
--save_interval 100
```

### 4. 保留原模型作为对照

```bash
# 先用原模型生成
python ../model_inference/Wan2.2-T2V-A14B.py

# 再用PINN插件生成
python inference_pinn.py --checkpoint_path ...

# 对比效果
```

## 📚 下一步

1. 阅读完整文档：`PINN_README.md`
2. 查看代码实现：
   - `diffsynth/models/pinn_operators.py` - 物理算子
   - `diffsynth/models/pinn_adapter.py` - 适配器
   - `diffsynth/pipelines/wan_video_pinn.py` - Pipeline
3. 实验不同场景和参数
4. 评估物理合理性

## ⚠️ 注意事项

1. **插件模式**：训练时原模型参数是冻结的，不用担心破坏原模型
2. **计算资源**：训练PINN插件比训练完整模型快得多（约10-20分钟）
3. **效果评估**：物理约束不是万能的，需要根据具体场景调整参数
4. **研究性质**：这是研究原型，建议在实际应用前充分测试

---

**祝您实验顺利！如有问题，请查阅 `PINN_README.md` 或提Issue。**
