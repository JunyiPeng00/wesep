# WavLMTasNet vs WavLMDynamicTasNet 对比

## 1. 总体差异概览

| 维度 | WavLMTasNet | WavLMDynamicTasNet |
|------|-------------|--------------------|
| **输入形式** | 两条独立波形：`mix_wav` + `ref_wav` | 一条拼接波形：`[enroll \| 静音 \| mix]`（或兼容两条波形） |
| **WavLM 调用方式** | mix 与 ref **各跑一次**完整 WavLM，得到两层特征 | **一次** WavLM 处理整段拼接，前 N 层 hybrid mask，后段仅 mix + QKb |
| **说话人嵌入** | MHFA 对 **ref 的全体层** 做加权 → 独立 d-vector | 前 N 层后对 **enroll 帧** 做时序平均 → 与 mix 同特征空间 |
| **说话人信息注入** | 在 10 ms 特征上 **乘性融合**（spk_fuse）后过 3×TCN | 在 **Transformer 内** 用 QKb 逐帧抑制 non-target keys，再层级加权 + TCN |
| **TasNet** | 共用：DeepEncoder → mask → DeepDecoder | 共用：仅对 mix 段编解码 |

---

## 2. 架构对比

### 2.1 WavLMTasNet

```
mix_wav ──► WavLM(mix) ──► layer_weighted_sum ──► 1×TCN ──► upsample ──► sep_feat_10ms
                                                                              │
ref_wav ──► WavLM(ref) ──► MHFA ──► spk_transform ──► spk_emb ─────────► spk_fuse ──► 3×TCN ──► mask_gen ──► mask
                                                                              │
enc_mix ◄─────────────────────────────────────────────────────────────────────┘
masked_enc = enc_mix * mask  →  decoder  →  est
```

- **Speaker**：ref 单独过 WavLM，再用 **MHFA**（多 head 注意力聚合）得到固定维 d-vector，与分离任务共享 WavLM 但 **不共享上下文**（ref 与 mix 的 attention 完全独立）。
- **Fusion**：在 10 ms 的 separator 特征上做 **乘性融合**（spk_emb 广播后与 sep_feat 逐元素乘），再经 3×TCN 出 mask。
- **计算**：WavLM 前向 **两次**（mix + ref），序列长度分别为 T_mix、T_ref。

### 2.2 WavLMDynamicTasNet

```
wav = [enroll | silence | mix]
        │
        ▼
   WavLM 单次前向（整段）
        │
   ┌────┴────┐
   │ 前 N 层  │  hybrid_attention_mask：enroll / 静音 / mix 段内自注意力，段间隔离
   └────┬────┘
        │
   spk_emb = mean(x[:, :enroll_end, :], dim=1)   ← Internalized Speaker (3.1)
   x_mix = x[:, mix_start:, :]                    ← Sequence Switch，丢弃 enroll
        │
   ┌────┴────┐
   │ 后 L-N 层 │  仅对 x_mix；每层 self-attention + QKb bias（s_t = σ(cos(h_t,e)), M_t = 1[s_t<τ], B_QKb 抑制 non-target keys）
   └────┬────┘
        │
   layer_weighted_sum(mix 各层输出) ──► mix_proj ──► TCN ──► mask_head ──► mask
        │
   enc_mix = encoder(mix_wav),  masked_enc = enc_mix * mask  →  decoder  →  est
```

- **Speaker**：与 mix **同一 backbone、同一前 N 层**，只对 enroll 对应帧做 **时序平均**，无 MHFA，无额外线性层，和后续 mix 特征空间一致（3.1 Internalized）。
- **Fusion**：说话人信息通过 **QKb** 注入到每层 self-attention（帧级抑制 non-target），而不是在 10 ms 特征上做乘性融合。
- **计算**：WavLM 前向 **一次**，序列长度为 T_enroll + T_silence + T_mix；后段只保留 T_mix 长序列，节省后 L−N 层的 enroll 部分计算。

---

## 3. 核心差异小结

| 项目 | WavLMTasNet | WavLMDynamicTasNet |
|------|-------------|--------------------|
| **Enroll/Ref 与 Mix 是否同一序列** | 否，两条独立序列 | 是，一条拼接序列 |
| **Speaker 特征空间** | MHFA 聚合 ref 的 WavLM 各层，可与 mix 特征有 domain gap | 前 N 层 enroll 帧平均，与 mix 完全同空间 |
| **Enroll 是否被 mix 干扰** | 不涉及（ref 单独编码） | 前 N 层用 hybrid mask 保证 enroll 看不到 mix |
| **说话人信息如何影响分离** | 在 10 ms 特征上乘性融合后 TCN | 在 Transformer 每层用 QKb 做帧级 key 抑制 |
| **WavLM 前向次数** | 2（mix + ref） | 1（整段） |
| **后段序列长度** | 无「后段」概念，mix 特征整段参与 | 后段仅 T_mix，enroll 被丢弃（Sequence Switch） |

---

## 4. 性能可能谁更好？原因简析

- **WavLMDynamicTasNet 更可能更好的理由**
  1. **Internalized speaker**：说话人嵌入与 mix 共用同一套浅层表征，和论文里说的“与识别任务特征空间对齐、减少与外部 SV 的 domain gap”一致，有利于在重叠段上更稳地认准目标说话人。
  2. **QKb 在 Transformer 内部做帧级选择**：在每一层都对「非目标」帧做 key 抑制，相当于在高层语义层面持续做 speaker-aware 的 attention 重分配，比只在 10 ms 特征上做一次乘性融合更细、更动态。
  3. **Enroll 不被 mix 污染**：前 N 层 hybrid mask 保证 enrollment 表示只来自 enroll 自身，有利于在强干扰下仍得到干净的 e。
  4. **计算与数据利用**：单次 WavLM 前向 + 后段只处理 T_mix，在长 mix、短 enroll 时更省算力；且一条样本里同时包含 enroll 与 mix，便于端到端学习「从 enroll 到 mix 的注意力偏好」。

- **WavLMTasNet 可能仍有优势的场景**
  1. **Ref 与 Mix 长度/域差异大**：ref 可以很长、来自不同采集条件，MHFA 对整段 ref 做聚合，可能在某些设定下更稳。
  2. **实现与调参成熟度**：若你处已有 WavLMTasNet 的充分调参与数据 pipeline，短期可能更容易复现或略优；WavLMDynamicTasNet 需要调 τ、N、hybrid_mask 等，超参更多。
  3. **数据形态**：若数据本来就是「分开的 ref + mix」且没有固定 enroll|silence|mix 的协议，WavLMTasNet 的接口更直接；WavLMDynamicTasNet 需要构造拼接或兼容接口。

**综合**：从设计动机（internalized speaker、QKb、hybrid mask、sequence switch）看，**WavLMDynamicTasNet 在重叠说话人、目标说话人建模和计算效率上更有潜力**；实际谁更好需在同一数据、同一协议（ref/enroll 长度、数据增强等）下做对比实验才能下结论。

---

## 5. 使用建议

- **优先尝试 WavLMDynamicTasNet**：当你的数据能提供「enroll | 静音 | mix」或可构造为单段拼接，且希望说话人嵌入与 mix 特征空间一致、并利用帧级 QKb 时。
- **保留 WavLMTasNet**：当 ref 与 mix 分离采集、ref 很长或形态多样，或你已有成熟的两路 WavLM + MHFA  pipeline 时，可作为对比基线或生产备选。
