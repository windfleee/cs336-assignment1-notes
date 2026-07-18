# BPE Tokenizer 从零读懂

> 结合 `tokenizer_implementation.md` 讲解思路 和 `tokenizer.py` 实际代码，
> 给初学者的逐段导读笔记。适合边读边讲。

---

## 一、开头：Tokenizer 到底在干什么？

大语言模型不认识"文本"，只认识"数字"。

你给它一句 `"Hello, world!"`，它要处理的是 `[42, 7, 103, ...]` 这样的 **整数序列**。

Tokenizer 就是这个**翻译官**：

```
文本  ——[分词]——> token ID 序列 ——[喂给]——> Transformer
文本  <——[拼接]—— token ID 序列 <——[吐出]—— Transformer
```

### 为什么不用单词直接做？

简单粗暴的想法是"一个词一个 ID"——但英文有几十万词，中文有几十万个词/字，词汇表太大，模型学不动。

另一个极端是"一个字符/字节一个 ID"——词汇表只有 256 种，但句子变成几十上百个 token，浪费 Transformer 的上下文窗口。

| 方案              | token 序列长度  | 词汇表大小      | 适合建模？ |
|:-----------------|:---------------|:---------------|:----------|
| 字符级（256 种）  | 长 ❌           | 小 ✅           | 否         |
| 词级（百万种）    | 短 ✅           | 大 ❌           | 否         |
| **BPE（50K 种）** | **中 ✅**       | **中 ✅**       | **是**     |

**BPE（Byte Pair Encoding）** 取的是中间路线：从 256 个字节出发，不断合并最高频的相邻字节对，直到词汇表达到想要的大小。

这种方法的好处是：
- **无 OOV（Out-of-Vocabulary）**：任何文本都能用基本字节表示，GPT-2 因此能处理所有语言的文本
- **常见词变成短 token**："the"、"and" 这种高频词很快合并成单个 token，而罕见词退回到更细粒度的片段
- **词汇表适中**：常用大小在 8K~100K 之间

---

## 二、Token 的三种形态（贯穿全文的核心概念）

同一个 token 在不同阶段以不同形式出现。这是所有后续代码的基础。

```
文本 "hello"
    │
    ▼ encode
[42, 7, 103]          ← token ID（整数，vocab 的 key）
    │  vocab 查表
    ▼
[b'he', b'l', b'lo']  ← token bytes（字节序列，vocab 的 value）
    │  Embedding 查表
    ▼
[v₀, v₁, v₂]          ← token vector（浮点数向量）
```

| 形态               | Python 类型     | 存储位置             | 在哪用                                      |
|:------------------|:----------------|:--------------------|:--------------------------------------------|
| **token ID**      | `int`           | `vocab` 的 key       | `encode` 产出、`decode` 输入、喂给模型       |
| **token bytes**   | `bytes`         | `vocab` 的 value     | 训练时决定"哪些字节该合并"、`decode` 时拼接   |
| **token vector**  | `float[d_model]`| Embedding 矩阵       | Transformer 内部计算                          |

### 一目了然的例子

假设 `"cat"` 经过 BPE 合并后变成了一个 token：

```python
token ID:    1523              # 词汇表中的编号
token bytes: b'cat'            # 3 个字节
token vector: [0.12, -0.34, ...]  # Embedding[1523] 查出来的 768 维向量
```

---

## 三、BPE 训练：train_bpe() 逐行读懂

### 3.1 整体流程（先看骨架）

```python
def train_bpe(input_path, vocab_size, special_tokens):
    # ① 初始化词汇表：256 个单字节 + 特殊 token
    # ② 预分词：把文本切分成"单词"
    # ③ 词频去重：相同单词只存一份，记下出现次数
    # ④ 循环合并：每次找最高频字节对，合并，更新
    return vocab, merges
```

### 3.2 初始化词汇表（第 1 步）

```python
vocab = {i: bytes([i]) for i in range(256)}
vocab.update({len(vocab) + i: token.encode('utf-8')
              for i, token in enumerate(special_tokens)})
merges = []
```

这就是 BPE 的"起点"：
- ID 0~255 对应单个字节 `b'\x00'` ~ `b'\xff'`
- `<|endoftext|>` 这样的特殊 token 领到 ID 256, 257, ...
- `merges` 列表一开始是空的，后面每合并一次就加一条

> **为什么从字节级开始？** 任何 UTF-8 文本都能表示为字节序列，这样不会有任何文本"查不到"。

### 3.3 预分词（第 2 步）

```python
# GPT-2 预分词正则（来源：openai/gpt-2）
 PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

这条正则把文本切成"单词级单位"，比如：

- `'t`、`'s`、`'re`、`'ll`、`'ve` → 单独保留（因为它们是英文缩写的后缀）
- ` hello`（含前导空格）→ 保留空格，确保单词边界信息不丢
- `123` → 数字序列
- `!!!` → 标点符号序列
- 多个空格单独成段

但注意，这里不是真正按"英文单词"切的——中文也会按 `\p{L}`（任何语言的字母）切分。

**特殊 token 如何处理？**

```python
special_tokens = sorted(special_tokens, key=len, reverse=True)  # 长的先匹配
parts = regex.split(f"({special_pattern})", full_text)
# 特殊 token 不进入预分词，直接作为整体保留
```

按长度降序排列是防止短的 token 把长的吃掉。`<|endoftext|>` 一定要比 `<|` 先匹配。

### 3.4 词频去重（关键优化！）

如果不做去重，原始实现要记录每个 pair 在哪个**位置**出现，大文件下位置列表会膨胀到 O(N²)。

**核心思路**：相同的单词存一次，记下它出现了几次。

| 数据结构        | 类型                      | 含义                     |
| :---------- | :---------------------- | :--------------------- |
| `words`     | `list[list[bytes]]`     | 唯一单词列表                 |
| `words_cnt` | `list[int]`             | 每个单词的出现次数              |
| `pair_cnt`  | `dict[tuple, int]`      | 字节对 → 总频次              |
| `pairpos`   | `dict[tuple, set[int]]` | 字节对 → 包含该 pair 的单词索引集合 |

代价是更新时要自己维护 `pair_cnt` 和 `pairpos`，但换来从 44 分钟降到秒级。

### 3.5 合并循环（第 4 步）

```python
for _ in range(cnt):                       # cnt = vocab_size - len(vocab)
    if not pair_cnt:
        break                              # 没有更多可合并的 pair

    # 找到出现次数最多的字节对（频次相同→字典序更大优先）
    best_pair = max(pair_cnt.items(), key=lambda p: (p[1], p[0]))[0]
    merges.append(best_pair)

    # 创建新 token，追加到 vocab 末尾
    new_token = best_pair[0] + best_pair[1]
    vocab[len(vocab)] = new_token

    # 只更新涉及 best_pair 的单词（通过 pairpos 定位）
    for idx in list(pairpos[best_pair]):
        word = words[idx]
        cnt = words_cnt[idx]

        # ① 移除旧 pair 计数（涉及此单词的每对相邻字节）
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair in pair_cnt:
                pair_cnt[pair] -= cnt
                if pair_cnt[pair] <= 0:       # 归零则彻底删除
                    del pair_cnt[pair]
                    del pairpos[pair]
                else:
                    pairpos[pair].discard(idx)

        # ② 执行合并：把 best_pair 合成一个 token
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                new_word.append(new_token)    # 两个字节合并为一个新 token
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        words[idx] = new_word

        # ③ 添加新 pair 统计（新单词中有哪些相邻对）
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            if pair not in pair_cnt:
                pair_cnt[pair] = cnt
                pairpos[pair] = {idx}
            else:
                pair_cnt[pair] += cnt
                pairpos[pair].add(idx)
```



---

## 四、Tokenizer 类：拼图全貌

### 4.1 __init__：两种查表结构

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab                 # id → bytes
        self.token_to_id = {v: k for k, v in vocab.items()}  # bytes → id（编码用）
        self.pair_to_merge = {pair: i for i, pair in enumerate(merges)} # pair → rank（O(1)找顺序）
```

为什么要存 `pair_to_merge`？

编码时，每个 word 内部需要"按 merge 创建顺序从小到大合并"——也就是最早创建的 pair 优先级最高。字典能在 O(1) 时间查到任意 pair 的 rank。

### 4.2 decode（ID → 文本）：最简单的一步

```python
def decode(self, ids):
    tokens = [self.id_to_token[id] for id in ids]   # 查 vocab
    return b''.join(tokens).decode('utf-8', errors='replace')  # 拼接 → 解码
```

简单说就是"反着走一遍"：
1. 每个 ID 去 vocab 里取回对应的 bytes
2. 把所有 bytes 拼在一起
3. 用 UTF-8 解码成字符串

`errors='replace'` 防止非法字节序列抛异常，它会用 `�` (U+FFFD) 替换坏字节。

### 4.3 encode（文本 → ID）：三步走

编码是最复杂的一步，分三阶段：

**第一步：预分词**（和训练时一模一样）

```python
# 按特殊 token 切分
special_tokens = sorted(special_tokens, key=len, reverse=True)
parts = regex.split(f"({special_pattern})", full_text)

# 对非特殊部分，用 GPT-2 正则切词
for token in regex.findall(PAT, part):
    words.append([bytes([b]) for b in token.encode('utf-8')])
```

**第二步：对每个 word 按 rank 合并**

训练时我们学到了所有合并规则。编码时每个 word 要尝试所有可能的合并，**按 rank 从小到大**（越早创建的越优先）：

```python
for word in words:
    while True:
        # 找当前 word 中 rank 最小的 pair
        best_pair = min(
            (p for p in pairs if p in self.pair_to_merge),
            key=lambda p: self.pair_to_merge[p],
            default=None
        )
        if best_pair is None:  # 没有可合并的了
            break
        word = merge_word(word, best_pair)  # while 保证跳过已合并位置
```

> **为什么 rank 小优先？** rank 小的 pair 在训练时先被合并，说明它更加高频/重要，编码时自然也应该先合。

**第三步：查 ID**

```python
ids = [self.token_to_id[token] for word in words for token in word]
```

### 4.4 encode_iterable：处理大文件的省内存方案

```python
def encode_iterable(self, iterable):
    for chunk in iterable:      # 可以传入 open('file.txt') 文件句柄
        for tid in self.encode(chunk):
            yield tid           # yield 逐个产出，内存恒定
```

`yield` 是 Python 生成器的关键字。调用者每拿一个 ID，函数才继续算下一个。这样无论文件多大，内存里始终只保存一小块数据。

### 4.5 from_files：从磁盘恢复

```python
@classmethod
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    # vocab 存为 JSON: {"0": [104], "1": [101], ...}
    raw = json.load(open(vocab_filepath))
    vocab = {int(k): bytes(v) for k, v in raw.items()}

    # merges 存为文本: 每行两个 token 的 UTF-8 编码
    merges = []
    for line in open(merges_filepath):
        a, b = line.strip().split(" ", 1)
        merges.append((a.encode('utf-8'), b.encode('utf-8')))

    return cls(vocab, merges, special_tokens)
```

`@classmethod` 意味着可以直接 `Tokenizer.from_files(...)` 调用，不用先创建实例。

注意 `split(" ", 1)` 只切第一刀——因为 token 本身可能包含空格（例如 `b' re'` 这种带前导空格的 token）。

---

## 五、训练 vs 编码：一张对照表

| 阶段     | 输入           | 做的事情                                     | 关键数据结构                                     |
|:--------|:--------------|:---------------------------------------------|:-----------------------------------------------|
| **训练** | 原始文本文件   | 从 256 字节开始，逐步合并最高频 pair，<br>直到 vocab 达到指定大小 | `pair_cnt`（频次表）<br>`pairpos`（位置索引）    |
| **编码** | 字符串         | 对每个预分词单元，按 rank 从小到大合并，查 ID   | `token_to_id`（bytes → ID）<br>`pair_to_merge`（pair → rank） |
| **解码** | ID 列表        | 查 vocab、拼 bytes、UTF-8 解码                | `id_to_token`（ID → bytes）                     |

---

## 六、踩坑记录（你可能也会碰到）

| 问题                           | 根因                                         | 怎么修                                       |
|:------------------------------|:---------------------------------------------|:---------------------------------------------|
| 大文件跑 44 分钟                | 存每个 pair 的每个位置 → 位置列表 O(N²)         | 词频去重（相同单词存一份 + 计数器）            |
| 三路平局 merge 顺序不同         | 降序处理重叠 pair 时左边被跳过                  | 升序 + 索引补偿                               |
| 词频去重后 KeyError            | 同 word 内 pair 多次出现，减计数时 set 已被删    | 移除空 set 前检查                             |
| 特殊 token 被拆开               | 短 token 在正则中先匹配                        | 按长度降序排列                                |
| `for` 内 `j += 1` 无效         | Python for 循环每轮重置 j                      | 改用 `while`                                 |
| macOS 不支持 rlimit            | 内存测试用 rlimit 跳过                         | 用 tracemalloc 替代                           |

---

## 七、推荐阅读顺序（给初学者）

如果你是从零开始，建议按这个顺序读/讲：

1. **Section 一**：理解 Tokenizer 是什么、为什么需要 BPE
2. **Section 二**：理解 Token 的三种形态（贯穿全文）
3. **Section 三（3.1~3.3）**：train_bpe 的前三步（初始化、预分词）
4. **Section 四（4.1~4.2）**：Tokenizer 类和最简单的 decode
5. **Section 三（3.4~3.5）**：词频去重和合并循环（最难的部分，放到中间）
6. **Section 四（4.3~4.5）**：encode、encode_iterable、from_files
7. **Section 五**：对照表总结
8. **Section 六**：踩坑记录

这样先建立"感性认识"，再深入到最复杂的合并逻辑，最后用总结收尾。

---


## 八、完整代码（tokenizer.py）

```python
"""
字节级 BPE 分词器训练、编码与解码。

参考资料:
  - 作业讲义 第2节
  - adapters.run_train_bpe (tests/adapters.py:565)
  - adapters.get_tokenizer (tests/adapters.py:542)
"""
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import regex

# ──────────────────────────────────────────────
# 第一部分: BPE 训练
# ──────────────────────────────────────────────

def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在给定的文本文件上训练一个字节级 BPE 分词器。

    参数:
        input_path: 训练数据 .txt 文件的路径。
        vocab_size: 最终词汇表的最大大小（包含初始的 256 字节词汇、
                    所有特殊 token、以及通过合并产生的所有新 token）。
        special_tokens: 需要作为完整 token 加入词汇表的字符串列表
                        （例如 ["<|endoftext|>"]）。训练时它们作为
                        硬边界 —— 合并操作不会跨越它们，且它们不参与
                        merge 统计。

    返回值:
        vocab:  dict[int, bytes]          token ID → token 字节序列
        merges: list[tuple[bytes, bytes]] BPE 合并列表，按创建顺序排列
    """
    # 初始化
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({len(vocab) + i: token.encode('utf-8')
                  for i, token in enumerate(special_tokens)})
    merges = []

    # 预分词
    full_text = ""
    with open(input_path, "rb") as f:
        full_text = f.read().decode("utf-8", errors="ignore")
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(regex.escape(t) for t in special_tokens)
    parts = []
    if special_tokens:
        parts = regex.split(f"({special_pattern})", full_text)
    else:
        parts = [full_text]
    words = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        if part not in special_tokens:
            for token in regex.findall(PAT, part):
                words.append([bytes([b]) for b in token.encode('utf-8')])

    # 词频去重
    tmpword = words.copy()
    words = []
    words_cnt = []
    pair_cnt = {}
    pairpos = {}
    che = {}
    for word in tmpword:
        if tuple(word) not in che:
            words.append(word)
            words_cnt.append(1)
            che[tuple(word)] = len(words) - 1
        else:
            idx = che[tuple(word)]
            words_cnt[idx] += 1
    for idx, word in enumerate(words):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in pair_cnt:
                pair_cnt[pair] = words_cnt[idx]
                pairpos[pair] = {idx}
            else:
                pair_cnt[pair] += words_cnt[idx]
                pairpos[pair].add(idx)

    # 合并循环
    cnt = vocab_size - int(len(vocab))
    for _ in range(cnt):
        if not pair_cnt:
            break
        best_pair = max(pair_cnt.items(), key=lambda p: (p[1], p[0]))[0]
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        for idx in list(pairpos[best_pair]):
            word = words[idx]
            cnt = words_cnt[idx]

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in pair_cnt:
                    pair_cnt[pair] -= cnt
                    if pair_cnt[pair] <= 0:
                        del pair_cnt[pair]
                        del pairpos[pair]
                    else:
                        pairpos[pair].discard(idx)

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                if pair not in pair_cnt:
                    pair_cnt[pair] = cnt
                    pairpos[pair] = {idx}
                else:
                    pair_cnt[pair] += cnt
                    pairpos[pair].add(idx)
    return vocab, merges


# ──────────────────────────────────────────────
# 第二部分: Tokenizer 类
# ──────────────────────────────────────────────

class Tokenizer:
    """
    BPE 分词器：将文本编码为整数 ID，并将 ID 解码回文本。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.pair_to_merge = {pair: i for i, pair in enumerate(merges)}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {int(k): bytes(v) for k, v in raw.items()}
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ", 1)
                    merges.append((a.encode('utf-8'), b.encode('utf-8')))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        full_text = text
        special_tokens = self.special_tokens
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(regex.escape(t) for t in special_tokens)
        parts = []
        if special_tokens:
            parts = regex.split(f"({special_pattern})", full_text)
        else:
            parts = [full_text]
        words = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:
            if part in special_tokens:
                words.append([part.encode('utf-8')])
            else:
                for token in regex.findall(PAT, part):
                    words.append([bytes([b]) for b in token.encode('utf-8')])

        for i in range(len(words)):
            word = words[i]
            while True:
                pairs = [(word[j], word[j + 1]) for j in range(len(word) - 1)]
                min_rank = float('inf')
                best_merge = None
                for pair in pairs:
                    if pair in self.pair_to_merge:
                        rank = self.pair_to_merge[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_merge = pair
                if best_merge is None:
                    break
                new_word = []
                j = 0
                while j < len(word):
                    if j+1 < len(word) and (word[j], word[j + 1]) == best_merge:
                        new_word.append(word[j] + word[j + 1])
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                word = new_word
            words[i] = word
        ids = []
        for word in words:
            for token in word:
                token_id = self.token_to_id.get(token)
                if token_id is not None:
                    ids.append(token_id)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[id] for id in ids]
        return b''.join(tokens).decode('utf-8', errors='replace')
```

---

