# 策略优化后验证指南

## 当前状态

✅ **配置优化完成**: `config.yaml` 已更新为优化后参数
⏳ **AI 模型训练中**: 预计完成时间约 22:50
⏳ **等待验证**: 训练完成后需运行反向验证测试效果

## 优化内容摘要

### 核心改进
1. **权重调整**: trend 40% / reversion 35% / volume 25%
2. **指标标准化**: MACD(12,26,9), RSI(14), ATR(14)
3. **AI 阈值提升**: 从 0.45 提升至 0.55（横盘市场）
4. **量价关系强化**: 量比要求从 1.55 提高至 2.0
5. **止盈优化**: 从 8% 提高至 10%，让利润奔跑

### 预期效果
- 胜率：从 0% → 50%+
- 平均收益：从 -2.68% → +2%+
- 盈亏比：从 0.00 → 1.5+

## 验证步骤

### 步骤 1: 确认训练完成

```bash
# 检查模型文件
.venv\Scripts\python.exe scripts\check_training_progress.py
```

**期望输出**:
```
[OK] 模型文件已存在
  文件大小：6-10 MB
  最后修改时间：[最近时间]
[OK] 模型加载成功
```

### 步骤 2: 单次反向验证（快速测试）

```bash
# 测试 2024-04-15 的表现（与优化前对比）
.venv\Scripts\python.exe scripts\simple_reverse_validation.py
```

**验证要点**:
- 信号数量是否合理（期望 3-5 个，而非优化前的 2 个）
- 胜率是否提升（期望 >= 50%）
- 平均收益是否转正（期望 >= 1.5%）

### 步骤 3: 多日期验证（全面测试）

```bash
# 测试 5 个不同日期的表现
.venv\Scripts\python.exe scripts\multi_date_validation.py
```

**测试日期**:
- 2024-03-15
- 2024-04-15
- 2024-05-15
- 2024-06-17
- 2024-07-15

**验证要点**:
- 多日期平均胜率（期望 >= 50%）
- 收益稳定性（标准差 < 5%）
- 不同市场环境下的适应性

### 步骤 4: 参数微调（如需要）

如果步骤 2 或 3 结果未达标：

```bash
# 使用优化器自动调优（15 轮，300 样本）
.venv\Scripts\python.exe main.py optimize --rounds 15 --samples 300
```

**优化目标**: `sharpe_pure`（夏普比率优先）

### 步骤 5: 生成验证报告

运行验证后，系统会生成 CSV 文件：
- `reverse_validation_YYYYMMDD_HHMMSS.csv` - 单次验证详细数据
- `multi_date_validation_YYYYMMDD_HHMMSS.csv` - 多日期验证详细数据

**分析方法**:
1. 打开 CSV 文件
2. 检查 `win_rate`（胜率）
3. 检查 `avg_pnl`（平均收益）
4. 检查 `profit_factor`（盈亏比）
5. 按 `ai_tier` 分组分析高置信度信号表现

## 成功标准

### 基础指标（必须达到）
- [ ] 胜率 >= 50%
- [ ] 盈亏比 >= 1.3
- [ ] 平均收益 >= 1.5%
- [ ] 最大单笔亏损 <= -5%

### 进阶指标（期望达到）
- [ ] 夏普比率 >= 1.5
- [ ] 高 AI 置信度胜率 > 低 AI 置信度
- [ ] 多日期测试胜率标准差 < 15%

### 失败处理

如果未达到基础指标：

1. **检查配置是否正确应用**
   ```bash
   # 查看 config.yaml 中的关键参数
   grep -A 3 "weights:" config.yaml
   grep "ai_threshold" config.yaml
   ```

2. **调整 AI 阈值**
   - 如果信号太多（>10 个/天）：提高 0.05
   - 如果信号太少（<2 个/天）：降低 0.05

3. **重新运行优化器**
   ```bash
   .venv\Scripts\python.exe main.py optimize --rounds 20 --samples 400
   ```

4. **检查数据质量**
   ```bash
   # 查看数据目录
   ls data/*.csv | head -5
   ```

## 命令参考

### 训练相关
```bash
# 训练 AI 模型
.venv\Scripts\python.exe main.py train-ai

# 检查训练进度
.venv\Scripts\python.exe scripts\check_training_progress.py

# 监控训练（实时）
.venv\Scripts\python.exe scripts\monitor_training.py
```

### 验证相关
```bash
# 单次反向验证
.venv\Scripts\python.exe scripts\simple_reverse_validation.py

# 多日期反向验证
.venv\Scripts\python.exe scripts\multi_date_validation.py

# 完整系统验证
.venv\Scripts\python.exe scripts\full_system_validation.py
```

### 优化相关
```bash
# 自动参数优化
.venv\Scripts\python.exe main.py optimize --rounds 15 --samples 300

# 批量回测
.venv\Scripts\python.exe main.py batch-test --num 50
```

### Web UI（可视化）
```bash
# 启动可视化平台
.venv\Scripts\python.exe main.py ui
```

访问：http://127.0.0.1:7860

## 常见问题

### Q1: 训练完成后胜率仍为 0%？

**可能原因**:
- AI 模型训练数据质量问题
- 市场状态分类不准确
- 选股逻辑存在 bug

**解决方案**:
1. 检查 `data/quant.log` 日志
2. 尝试降低 AI 阈值 0.1 重新测试
3. 手动检查几只股票的数据质量

### Q2: 信号数量过少（<1 个/天）？

**可能原因**:
- AI 阈值过高
- 选股条件过严
- 测试日期市场整体无信号

**解决方案**:
1. 降低 AI 阈值 0.05-0.1
2. 降低 `vol_up_ratio` 至 1.8
3. 更换测试日期

### Q3: 信号数量过多（>20 个/天）？

**可能原因**:
- AI 阈值过低
- 选股条件过松

**解决方案**:
1. 提高 AI 阈值 0.1
2. 提高 `vol_up_ratio` 至 2.5
3. 缩短 `min_hold_days` 至 2

### Q4: 平均收益为负但胜率高？

**可能原因**:
- 止盈过早，亏损单持有过久
- 盈亏比不合理

**解决方案**:
1. 提高 `take_profit_pct` 至 0.12-0.15
2. 降低 `trail_atr_mult` 至 1.5
3. 检查 `breakeven_trigger` 是否合理

## 日志位置

- **主日志**: `data/quant.log`
- **训练日志**: 查看 `scripts/monitor_training.py` 输出
- **验证日志**: 查看反向验证脚本输出

## 备份与恢复

### 备份配置
```bash
# 备份当前配置
cp config.yaml config_backup_$(date +%Y%m%d).yaml
```

### 恢复配置
```bash
# 恢复到优化前配置
cp config_backup_20260317.yaml config.yaml
```

### 备份模型
```bash
# 备份当前模型
cp models\alpha_lgbm.txt models\alpha_lgbm_backup_$(date +%Y%m%d).txt
```

## 下一步

1. ✅ 等待 AI 训练完成
2. ⏳ 运行单次反向验证
3. ⏳ 分析结果，如未达标则调整参数
4. ⏳ 运行多日期验证
5. ⏳ 生成最终验证报告
6. ⏳ 准备实盘测试

---

**文档更新时间**: 2026-03-17 22:50
**当前状态**: AI 训练中
**下一步**: 检查训练进度，准备验证测试
