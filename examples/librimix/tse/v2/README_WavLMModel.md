## WavLM-TSE 模型说明

本文档面向 `examples/librimix/tse/v2` 中基于 WavLM 的目标说话人语音分离模型，聚焦以下内容：
- 如何生成可用的 WavLM 转换模型
- `WavLMTasNet` 与 `WavLMDynamicTasNet` 的结构设计
- 配置文件与模型实现的对应关系
- 训练、推理与最小验证方式

如需查看 LibriMix 数据准备与完整 stage 1~6 流程，请参考同目录 `README.md`。

---

### 1. 第一步：生成转换后的 WavLM 模型

本项目中的 WavLM 前端不直接加载 Hugging Face 原始权重，而是加载转换后的 `.pth` 文件，格式为：

```python
{"config": ..., "state_dict": ...}
```

本目录已提供转换脚本 `download_model.py`。建议先在 CPU 环境完成下载与转换，再进行训练或推理，以减少运行时下载和格式不一致带来的不确定性。

执行命令：

```bash
cd examples/librimix/tse/v2
conda activate wesep
python download_model.py
```

默认会在当前目录生成 `convert/`，并输出：
- `convert/wavlm_base_plus.hf.pth`
- `convert/wavlm_large.hf.pth`

如仅需校验模型文件是否完整可用，可执行：

```bash
python download_model.py verify
```

完成后，请将配置文件中的 `wavlm_ckpt` 修改为真实路径，例如：

```yaml
wavlm_ckpt: /path/to/examples/librimix/tse/v2/convert/wavlm_base_plus.hf.pth
```

需要修改的位置：
- `confs/wavlm_tasnet.yaml`
- `confs/wavlm_dynamic_tasnet.yaml`

---

### 2. 代码入口

- `wesep/wesep/models/wavlm_tasnet.py`：`WavLMTasNet`
- `wesep/wesep/models/wavlm_dynamic_tasnet.py`：`WavLMDynamicTasNet`
- `wesep/wesep/modules/wavlm_frontend/frontend.py`：`HuggingfaceFrontendWavLM`
- `wesep/wesep/modules/ssl_backend/MHFA.py`：`SSL_BACKEND_MHFA`
- `wesep/wesep/modules/common/speaker.py`：`SpeakerTransform`、`SpeakerFuseLayer`

建议先阅读 `frontend.py` 与两个模型文件，再结合 YAML 配置理解训练入口；这样最容易建立“配置 -> 模型实例化 -> 前向路径”的对应关系。

---

### 3. WavLM 前端输出约定

`HuggingfaceFrontendWavLM.forward(input_wav, input_lengths)` 的输入输出如下：

- 输入：`input_wav` 为 `[B, T]`，16kHz 单声道波形
- 输出：`layer_reps` 为 `[B, C, T_frames, L]`

其中：
- `B`：batch size
- `C`：WavLM 隐表示维度
- `T_frames`：约 20 ms 分辨率的帧长
- `L`：层数，包含 pre-transformer 表示

后续所有 WavLM-TSE 模型都基于这组层级特征完成三类操作：
- 对层表示做加权或聚合
- 从 enrollment 中提取说话人表征
- 结合时域 encoder 特征预测分离 mask

从工程实现角度看，这种设计把“说话人建模”和“时域重建”明确分离：
- WavLM 负责提供高质量的语音表征
- encoder/decoder 负责保持时域重建能力
- 中间的融合层负责将说话人先验注入分离路径

---

### 4. `WavLMTasNet` 结构

接口：

```python
forward(mix_wav, ref_wav) -> est_wav
```

输入输出：
- `mix_wav`：`[B, T]`
- `ref_wav`：`[B, T_ref]`
- `est_wav`：`[B, T]`

#### 4.1 结构分解

1. 时域编码器
- 使用 `Conv1d` 将混合语音映射到 encoder 域，得到 `[B, encoder_dim, T_10]`

2. 共享 WavLM 前端
- 混合语音生成 `ssl_mix`：`[B, C, T_20, L]`
- 参考语音生成 `ssl_ref`：`[B, C, T_20_ref, L]`

3. 混合侧建模
- 对 `ssl_mix` 做可学习的 layer-wise 加权求和
- 在 20 ms 分辨率上通过 TCN 建模上下文
- 投影到 `sep_tcn_channels`
- 通过可学习上采样对齐到 10 ms 分辨率

4. 说话人侧建模
- 使用 `SSL_BACKEND_MHFA` 对 `ssl_ref` 聚合得到 speaker embedding
- 使用 `SpeakerTransform` 对 embedding 做分布对齐与非线性变换

5. 说话人融合
- 使用 `SpeakerFuseLayer(FiLM)` 将 speaker embedding 注入分离特征

6. mask 预测与解码
- 预测 encoder 域 mask
- 与混合语音 encoder 特征逐点相乘
- 使用 `ConvTranspose1d` 解码回时域波形

#### 4.2 设计特点

- 混合语音路径与 enrollment 路径职责明确，便于调试和替换模块
- WavLM 负责语义与说话人相关表征，TasNet 负责时域重建
- 对于使用独立 enrollment 的目标说话人分离场景，这种结构更直接，也更稳定

---

### 5. `WavLMDynamicTasNet` 结构

接口：

```python
forward(mix_wav, enroll_wav, return_embedding=False)
```

输入输出：
- `mix_wav`：`[B, T_mix]`
- `enroll_wav`：`[B, T_enroll]`
- 默认输出：`est_wav`，形状为 `[B, T_mix]`
- 当 `return_embedding=True` 时，额外返回 `spk_emb`

#### 5.1 结构分解

1. 输入拼接
- 内部将输入组织为 `enroll + silence + mix`

2. hybrid attention / hybrid mask
- 在 WavLM encoder 早期层，对 enrollment 与 mixture 构建结构化注意力约束

3. 说话人嵌入提取
- 对 enrollment 段相关隐藏层堆叠后，使用 `SSL_BACKEND_MHFA` 提取 speaker embedding

4. QKB 引导
- 在 mixture 路径的后续层，用 speaker embedding 与 mixture 特征计算连续注意力偏置

5. mask 与解码
- 输出 encoder 域 ReLU mask
- 与混合语音编码特征相乘
- 再解码回时域

#### 5.2 设计特点

- 将 enrollment 与 mixture 放入统一的上下文建模框架
- 强化了“目标说话人条件”对注意力与 mask 的显式约束
- 更适合研究 speaker-guided attention、hybrid mask、动态条件分离等方向

相较于 `WavLMTasNet`，该模型的表达能力更强，但路径也更复杂，对配置、显存与训练稳定性更敏感。

---

### 6. 配置文件对应关系

#### 6.1 `confs/wavlm_tasnet.yaml`

对应模型：`WavLMTasNet`

#### 6.2 `confs/wavlm_dynamic_tasnet.yaml`

对应模型：`WavLMDynamicTasNet`

---

### 7. 训练与推理调用

在 `examples/librimix/tse/v2` 下执行：

训练：

```bash
cd examples/librimix/tse/v2
bash run.sh --stage 3 --stop_stage 3
```

推理：

```bash
bash run.sh --stage 5 --stop_stage 5
```

切换到 dynamic 版本：

```bash
bash run.sh --config confs/wavlm_dynamic_tasnet.yaml --stage 3 --stop_stage 3
```

这里的调用逻辑很简单：
- `run.sh` 负责组织配置、路径和 stage
- `train.py` 负责实例化模型并训练
- `infer.py` 负责加载 checkpoint 做目标说话人分离
