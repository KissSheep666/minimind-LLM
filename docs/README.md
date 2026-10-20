# MiniMind 学习仓库

> 借助 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 理解大语言模型的完整链路与基本原理。

## 这是什么

这个仓库是**学习用工作区**,不是 MiniMind 的官方镜像。代码来自上游仓库(Apache 2.0 协议,`LICENSE` 原样保留),文档部分为本站原创,记录的是从零开始跑通、读懂这个小模型的全过程。

上游项目做的事:**从 0 用原生 PyTorch 训练一个 64M 参数的语言模型**,单卡 3090 约 2.3 小时、约 3 元成本。它不依赖 `transformers` / `trl` / `peft` 的高层封装,Attention、RoPE、MoE、DPO、GRPO 全部手写。

## 文档地图

建议按顺序读,每一篇都可以独立查阅。

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [00 · Python 速成](00-Python速成.md) | 只补读这份代码真正需要的那几样语法 | 打开源码前先扫一眼 |
| [01 · 项目评估](01-项目评估.md) | 这项目值不值得学、有什么坑 | **先读这篇** |
| [02 · 快速跑通](02-快速跑通.md) | 在本地 CPU 上让它开口说话 | 动手第一步 |
| [03 · 模型结构](03-模型结构.md) | 292 行模型代码逐模块拆解 | 跑通之后 |
| [04 · 训练链路](04-训练链路.md) | 数据、Pretrain、SFT、RL 到底在做什么 | 读懂结构之后 |
| [05 · 踩坑速查](05-踩坑速查.md) | 环境、运行、训练三类的坑与解法 | 卡住时查 |

## 十分钟快速开始

```bash
# 1. 建环境(必需:本机全局 numpy 与项目锁定版本冲突)
python -m venv .venv
source .venv/Scripts/activate

# 2. 装最小依赖(必需换清华源:PyPI 与阿里云源在本机超时)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.6 numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 下载 128MB 的模型权重
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# 4. 跑起来(输入 1 进入手动聊天)
python eval_llm.py --load_from ./minimind-3 --device cpu
```

> 本仓库的 `eval_llm.py` 已针对 CPU 做过一处精度修正(原版强制 fp16,在 CPU 上会报错),详见 [02 · 快速跑通](02-快速跑通.md)。

## 参考环境

本仓库的实测环境:

| 项 | 值 |
|---|---|
| CPU | Intel i5-1240P,12 核 16 线程 |
| 内存 | 16 GB |
| GPU | 无 NVIDIA 独显(仅 Intel Iris Xe 核显) |
| Python | 3.12.10 |
| 系统 | Windows 11 Pro 21H2 |
| 网络 | 系统代理 `127.0.0.1:7890`,HuggingFace 走代理可达,PyPI 需换清华源 |

**没有 GPU 也能学**:64M 模型的 CPU 推理和 LoRA 微调都跑得动。只有完整复现 Pretrain + SFT 才需要租卡。

## 学习路线

```mermaid
flowchart LR
    A[02 跑通推理] --> B[00 补 Python]
    B --> C[03 读懂 292 行模型]
    C --> D[04 读懂训练链路]
    D --> E[阶段 4:LoRA / RL / 租卡复现]
```

## 代码来源与协议

- 上游仓库:<https://github.com/jingyaogong/minimind>
- 上游协议:Apache License 2.0,见 [LICENSE](../LICENSE)
- 本仓库对上游代码的唯一改动:`eval_llm.py` 的 CPU 精度修正
- 文档与实验记录:原创

如果你要向上游同步更新:

```bash
git remote add upstream https://github.com/jingyaogong/minimind.git
git fetch upstream
git merge upstream/master
```
