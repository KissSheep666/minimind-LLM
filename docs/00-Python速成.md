# 00 · Python 速成:只为读懂这份代码

> 不需要先学完一整本 Python 教程。这份代码只用到下面这些语法,按需查、边读边补即可。

---

## 1. 函数与默认参数

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    ...
```

- `start_step=0` 是默认值,调用时不传就用 0。
- 调用时用名字传参:`train_epoch(epoch, loader, len(loader), 0, wandb)`。

---

## 2. 类、`self`、继承

这是整个仓库最核心的写法:

```python
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()          # 必须先调用父类的构造函数
        self.q_proj = nn.Linear(...) # 挂在自己身上的属性
    def forward(self, x):
        ...
```

要点:

- `class Attention(nn.Module)` 表示 **Attention 继承自 nn.Module**。继承之后它才能被 PyTorch 当作"一层网络"来管理参数、跑反向传播。
- `self` 就是"这个对象自己"。`self.q_proj` 表示把 `q_proj` 存到对象上,后面 `self.q_proj(x)` 才能取出来用。
- `super().__init__()` 调用父类(`nn.Module`)的初始化。**漏了这行,参数就不会被注册,训练时梯度更新不到。**

**怎么读**:看到 `class Xxx(nn.Module)`,先找它的 `__init__`(定义了哪些组件)和 `forward`(数据怎么流过这些组件)。这两个方法就是这一层的全部。

---

## 3. `*args` 和 `**kwargs`

```python
def forward(self, input_ids, attention_mask=None, ..., **kwargs):
```

- `**kwargs` 把所有"没显式列出的关键字参数"打包成一个字典。
- 仓库里到处都是,原因是它在兼容 `transformers` 的调用约定——框架会传进来一些我们暂时用不到的参数,用 `**kwargs` 兜住就不会报错。

看到 `**kwargs` 不用慌,**它通常意味着"这些参数我们不用管"**。

---

## 4. 装饰器 `@`

```python
@property
@torch.no_grad()
@torch.inference_mode()
```

- 装饰器写在函数上面一行,作用是在不修改函数本身的前提下给它加行为。
- `@torch.no_grad()`:告诉 PyTorch "这段不记录梯度"。推理时用它省显存、加速。
- `@torch.inference_mode()`:更严格的推理模式,`generate()` 用的就是它。

---

## 5. 切片与索引

这是读张量代码的必备技能。

```python
logits[..., :-1, :]      # 最后一维取全部,倒数第二维去掉最后一个,其余全部
labels[..., 1:]          # 倒数第二维从第 1 个开始到结尾(即去掉第一个)
input_ids[:, past_len:]  # 第一维全部,第二维从 past_len 开始
x[..., : x.shape[-1] // 2]  # 最后一维取前一半
```

- `:` 表示"这一维全要"。
- `...`(省略号)表示"前面所有维度我都不关心,原样保留"。
- `-1` 表示"最后一维"。`shape[-1]` 就是最后一维的大小。
- `+1` / `-1` 在索引里表示"跳过首/尾一个"。**下一个词预测的错位就是靠它实现的**,见 [04 · 训练链路](04-训练链路.md)。

---

## 6. 列表推导式

```python
self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
```

读法:**"对 `l in range(8)` 里的每个 l,都造一个 `MiniMindBlock(l, config)`,collect 成一个列表"**。等价于:

```python
layers = []
for l in range(8):
    layers.append(MiniMindBlock(l, config))
```

仓库里还常见带条件的推导式:

```python
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], ...)
```

**"把符合 if 条件的那些项的 aux_loss 加起来"**。

---

## 7. 三元表达式

```python
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
```

读作:"如果 `args.dtype == "bfloat16"` 就用 `torch.bfloat16`,否则用 `torch.float16`"。

---

## 8. `with` 语句

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
```

`with` 的作用是"进入一个上下文,出来时自动收尾"。这里 `autocast_ctx` 控制混合精度,缩进内的计算自动用半精度,出了缩进自动恢复。

---

## 9. 常见数值写法

| 写法 | 含义 |
|---|---|
| `1e-6` | 0.000001 |
| `5e-4` | 0.0005 |
| `-100` | PyTorch 里"这个位置不算 loss"的特殊标记 |
| `None` | 空,表示"没有" |
| `x.float()` / `x.half()` | 转成 32 位 / 16 位浮点 |
| `.contiguous()` | 把张量在内存里排成连续块(某些操作的前置要求) |

---

## 10. 读懂一份代码的最小方法

1. **先看入口**(`if __name__ == "__main__":`)——命令行参数定义了所有可调的东西。
2. **再看数据形状**。在关键位置插 `print(x.shape)`,跑一次,形状对上了逻辑就通了。
3. **不要逐行翻译成中文**。抓住每个函数"输入什么、输出什么、中间做了一次什么变换"就够了。

---

## 自检

读完这份文档,你应该能回答:

- `class Attention(nn.Module)` 里的 `super().__init__()` 为什么要写?
- `labels[..., 1:]` 比 `labels` 少了什么?
- `[f(i) for i in range(8)]` 等价于什么循环?

答不上来就回去看对应小节,然后直接打开 `model/model_minimind.py` 试读 `RMSNorm`(只有 11 行)。
