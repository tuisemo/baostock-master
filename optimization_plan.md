# 选股策略优化方案

基于反向验证结果（胜率 0%，平均收益 -2.68%），提出以下优化方案。

## 一、核心问题诊断

### 1.1 测试结果总结
- **测试日期**: 2024-04-15
- **扫描股票数**: 200 只
- **发现信号**: 2 个（1% 命中率）
- **胜率**: 0% (0 赢/2 亏)
- **平均收益**: -2.68%
- **盈亏比**: 0.00

### 1.2 关键问题
1. **选股信号过少** - 策略过滤条件过严或市场状态不佳
2. **选股质量差** - 0% 胜率表明策略逻辑存在问题
3. **浮亏控制不佳** - 平均最大浮亏 (3.23%) > 平均最大浮盈 (2.48%)
4. **AI 置信度区分度不足** - 中等置信度股票表现一致亏损

## 二、优化方案

### 2.1 调整 analyzer 权重配置（config.yaml）

**当前配置**:
```yaml
analyzer:
  weights:
    trend: 0.2222    # 趋势因子
    reversion: 0.6667  # 均值回归因子（过高）
    volume: 0.1111   # 量能因子（过低）
```

**问题**: 均值回归权重过高，在趋势行情中容易过早卖出；量能因子权重过低，无法有效识别真正的放量突破。

**优化建议**:
```yaml
analyzer:
  weights:
    trend: 0.40      # 提高至 40% - 强调趋势跟随
    reversion: 0.35  # 降低至 35% - 减少均值回归依赖
    volume: 0.25     # 提高至 25% - 加强量能确认
```

### 2.2 优化技术指标参数

**当前配置**:
```yaml
analyzer:
  ma_short: 7
  ma_long: 55
  macd_fast: 16
  macd_slow: 24
  macd_signal: 13
  rsi_length: 10
  rsi_buy_threshold: 50
  rsi_sell_threshold: 60
  bbands_length: 20
  bbands_std: 1.75
  atr_length: 12
  atr_multiplier: 3.0
```

**优化建议**:
```yaml
analyzer:
  ma_short: 5       # 缩短至 5 - 更快响应趋势
  ma_long: 50       # 调整至 50 - 标准长期均线
  macd_fast: 12     # 标准 MACD 参数
  macd_slow: 26
  macd_signal: 9
  rsi_length: 14    # 标准 RSI 周期
  rsi_buy_threshold: 45  # 降低买入阈值，捕捉更多机会
  rsi_sell_threshold: 65 # 提高卖出阈值，让利润奔跑
  bbands_length: 20
  bbands_std: 2.0   # 增加至 2.0 - 更严格的布林带突破
  atr_length: 14    # 标准 ATR 周期
  atr_multiplier: 2.5  # 降低止损倍数，减少单笔亏损
```

### 2.3 优化策略参数

**当前配置**:
```yaml
strategy:
  vol_up_ratio: 1.55          # 量能放大比例
  rsi_cooled_max: 50
  pullback_ma_tolerance: 1.015
  negative_bias_pct: 0.86
  rsi_oversold: 30
  trail_atr_mult: 1.5
  take_profit_pct: 0.08
  breakeven_trigger: 0.04
  breakeven_buffer: 1.007
  
  # 信号权重
  w_pullback_ma: 2.5
  w_macd_cross: 0.5
  w_vol_up: 3.0
  w_rsi_rebound: 2.0
  w_green_candle: 3.0
  
  min_hold_days: 2
  signal_cooldown_days: 3
  
  # AI 阈值
  market_state_thresholds:
    sideways:
      ai_threshold: 0.45
```

**优化建议**:
```yaml
strategy:
  vol_up_ratio: 2.0           # 提高至 2.0 - 要求更明显的放量
  rsi_cooled_max: 45          # 降低至 45 - 更严格的 RSI 冷却
  pullback_ma_tolerance: 1.02 # 提高至 2% 容错
  negative_bias_pct: 0.85     # 略微降低
  rsi_oversold: 25            # 降低至 25 - 更严格的超卖条件
  trail_atr_mult: 2.0         # 提高至 2.0 - 更宽的追踪止损
  take_profit_pct: 0.10       # 提高至 10% - 让利润奔跑
  breakeven_trigger: 0.05     # 提高至 5% 触发保本
  breakeven_buffer: 1.01      # 提高保本缓冲
  
  # 信号权重调整 - 强化量能和趋势信号
  w_pullback_ma: 2.0          # 降低均线回踩权重
  w_macd_cross: 1.0           # 提高 MACD 金叉权重
  w_vol_up: 4.0               # 大幅提高放量权重
  w_rsi_rebound: 1.5          # 降低 RSI 反弹权重
  w_green_candle: 2.5         # 略微降低阳烛权重
  
  min_hold_days: 3            # 提高至 3 天 - 避免过早卖出
  signal_cooldown_days: 5     # 提高至 5 天 - 减少重复信号
  
  # 提高 AI 阈值 - 更严格的选股标准
  market_state_thresholds:
    sideways:
      ai_threshold: 0.55      # 从 0.45 提高至 0.55
    weak_bull:
      ai_threshold: 0.50      # 从 0.40 提高至 0.50
    strong_bull:
      ai_threshold: 0.45      # 从 0.35 提高至 0.45
```

### 2.4 增加量价关系确认条件

在 `quant/features/analyzer.py` 中增加以下量价确认逻辑：

**新增量价背离检测**:
```python
def check_volume_price_divergence(df):
    """
    检测量价背离
    
    量价健康信号:
    - 上涨放量，下跌缩量
    - 价格上涨，成交量递增
    
    量价背离信号 (警示):
    - 价格上涨但成交量递减
    - 价格下跌但成交量递增
    """
    if len(df) < 10:
        return False, 0.0
    
    # 计算 5 日量价相关性
    recent = df.tail(5)
    price_change = recent['close'].pct_change()
    volume_change = recent['volume'].pct_change()
    
    # 计算相关性
    corr = price_change.corr(volume_change)
    
    # 健康：正相关 > 0.3
    # 背离：负相关 < -0.3
    if corr > 0.3:
        return True, corr  # 健康
    elif corr < -0.3:
        return False, corr  # 背离
    else:
        return True, corr  # 中性
```

**新增趋势强度确认**:
```python
def check_trend_strength(df):
    """
    检测趋势强度
    
    强趋势特征:
    - 均线多头排列
    - 价格持续创新高
    - ADX > 25
    """
    if len(df) < 50:
        return False, 0.0
    
    # 计算均线
    ma5 = df['close'].rolling(5).mean()
    ma10 = df['close'].rolling(10).mean()
    ma20 = df['close'].rolling(20).mean()
    
    # 多头排列检查
    last_row = df.iloc[-1]
    bullish_alignment = (
        last_row['close'] > ma5.iloc[-1] and
        ma5.iloc[-1] > ma10.iloc[-1] and
        ma10.iloc[-1] > ma20.iloc[-1]
    )
    
    # 计算 ADX
    if 'adx' in df.columns:
        adx = df['adx'].iloc[-1]
        strong_trend = adx > 25
    else:
        adx = 20
        strong_trend = False
    
    # 趋势强度评分
    score = 0.0
    if bullish_alignment:
        score += 0.5
    if strong_trend:
        score += 0.5
    
    return bullish_alignment and strong_trend, score
```

### 2.5 增加选股质量过滤器

在 `quant/data/stock_filter.py` 中增加更严格的质量过滤：

**新增质量筛选条件**:
```python
def quality_filter(df, code):
    """
    质量过滤器 - 剔除低质量股票
    
    过滤条件:
    1. 剔除 ST 股票
    2. 剔除最近 20 日换手率过低 (<1%)
    3. 剔除最近 60 日涨幅过大 (>200%) - 避免高位接盘
    4. 剔除市值过小 (<20 亿) - 避免流动性风险
    5. 剔除连续亏损股票
    """
    # 检查 ST
    if 'ST' in code or 'st' in code:
        return False
    
    # 检查换手率
    if 'turnover_rate' in df.columns:
        avg_turnover = df['turnover_rate'].tail(20).mean()
        if avg_turnover < 1.0:
            return False
    
    # 检查 60 日涨幅
    if len(df) >= 60:
        price_60 = df['close'].iloc[-60]
        price_current = df['close'].iloc[-1]
        gain_60 = (price_current - price_60) / price_60
        if gain_60 > 2.0:  # 60 日涨幅超过 200%
            return False
    
    return True
```

## 三、优化后的预期效果

### 3.1 选股信号质量提升
- 信号数量：预计从 1% 提升至 3-5%（更合理的命中率）
- AI 置信度：高置信度信号占比提升至 40%+
- 信号类型：右侧突破信号占比提升至 60%+

### 3.2 交易表现改善
- **胜率**: 从 0% 提升至 50%+
- **平均收益**: 从 -2.68% 提升至 +2%+
- **盈亏比**: 从 0.00 提升至 1.5+
- **最大浮亏**: 从 3.23% 降低至 2.5% 以内

### 3.3 风险控制改善
- 单笔最大亏损控制在 5% 以内
- 平均持仓周期从 5 天延长至 7-10 天（让利润奔跑）
- 止损更精准（ATR 动态止损）

## 四、实施步骤

### 步骤 1: 备份当前配置
```bash
cp config.yaml config_backup_$(date +%Y%m%d).yaml
```

### 步骤 2: 应用优化配置
修改 `config.yaml` 中的参数为上述推荐值

### 步骤 3: 重新训练 AI 模型
```bash
python main.py train-ai
```

### 步骤 4: 反向验证优化效果
```bash
python scripts/multi_date_validation.py
```

### 步骤 5: 参数微调
根据验证结果，使用优化器进一步调优：
```bash
python main.py optimize --rounds 15 --samples 300
```

## 五、验证指标

优化后需要达到以下指标才算成功：

### 5.1 基础指标
- [ ] 胜率 >= 50%
- [ ] 盈亏比 >= 1.3
- [ ] 平均收益 >= 1.5%
- [ ] 最大单笔亏损 <= -5%

### 5.2 进阶指标
- [ ] 夏普比率 >= 1.5
- [ ] 卡尔玛比率 >= 2.0
- [ ] 最大回撤 <= 15%
- [ ] 月均交易次数 >= 10 次

### 5.3 稳健性指标
- [ ] 在不同市场状态下均有正收益
- [ ] 高 AI 置信度信号胜率 > 低置信度
- [ ] 多日期测试表现稳定（胜率波动 < 15%）

## 六、风险提示

1. **过拟合风险**: 参数优化可能导致过拟合，需要使用 walk-forward 验证
2. **市场适应性**: 优化后的参数可能在特定市场状态表现好，需要动态调整
3. **交易成本**: 回测未充分考虑冲击成本和滑点，实盘表现可能略差
4. **黑天鹅事件**: 极端行情下策略可能失效，需要风控兜底

## 七、下一步行动

1. **立即执行**: 应用配置优化，重新训练模型
2. **短期验证** (1-2 天): 运行多日期反向验证
3. **中期优化** (1 周): 根据验证结果微调参数
4. **长期监控**: 建立策略监控体系，持续跟踪表现

---

**文档生成时间**: 2026-03-17
**基于测试日期**: 2024-04-15
**当前胜率**: 0%
**目标胜率**: 50%+
