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
    # TODO: 实现
    #初始化
    vocab = {i: bytes([i]) for i in range(256)}  # 初始化词汇表为所有单字节
    vocab.update({len(vocab) + i: token.encode('utf-8') for i, token in enumerate(special_tokens)})  # 添加特殊 token
    merges = []

    #预分词
    full_text = ""
    with open(input_path, "rb") as f:
        full_text = f.read().decode("utf-8", errors="ignore") #用二进制读入字符串
    # 1. 构造匹配所有特殊 token 的模式，用 | 连接
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pattern = "|".join(regex.escape(t) for t in special_tokens)
    # 2. 按特殊 token 切分，括号保留分隔符本身
    parts = []
    if special_tokens:
        parts = regex.split(f"({special_pattern})", full_text)
    else:
        parts = [full_text]
    # "Héllò hôw <|endoftext|> are ü?" → ["Héllò hôw ", "<|endoftext|>", " are ü?"]
    # 3. 对每个部分分别处理
    words = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for part in parts:
        #if part in special_tokens:
            # 特殊 token: 直接加入（整个作为 bytes 序列）
            #words.append([part.encode('utf-8')])
        if part not in special_tokens:
            # 普通文本: 正常预分词
            for token in regex.findall(PAT, part):
                words.append([bytes([b]) for b in token.encode('utf-8')])

    tmpword = words.copy()
    words = [] #存去重后的单词
    words_cnt = [] #存每个单词出现的次数
    pair_cnt = {} #存每个字节对出现的次数
    pairpos = {} #存每个字节对出现的索引
    che = {} # 存单词索引
    #初始化
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
    
    cnt = vocab_size - int(len(vocab))  # 计算需要的合并次数
    for _ in range(cnt):
        if not pair_cnt:
            break  # 没有更多的字节对可以合并
        # 找到出现次数最多的字节对
        best_pair = max(pair_cnt.items(), key=lambda p: (p[1], p[0]))[0]
        merges.append(best_pair)

        # 创建新 token
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        #只更新涉及 best_pair 的单词
        #更新 pair_cnt 和 pairpos
        for idx in list( pairpos[best_pair] ):
            word = words[idx]
            cnt = words_cnt[idx]
            # 移除旧的字节对计数
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in pair_cnt:
                    pair_cnt[pair] -= cnt
                    if pair_cnt[pair] <= 0:
                        del pair_cnt[pair]
                        del pairpos[pair]
                    else:
                        pairpos[pair].discard(idx)
            # 创建新单词
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
            # 更新新的字节对计数
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

    推荐接口（见作业讲义 第 2.6 节）。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        从词汇表和合并列表构造分词器。

        参数:
            vocab:  dict[int, bytes]          token ID → token 字节序列
            merges: list[tuple[bytes, bytes]] BPE 合并列表，按创建顺序排列
            special_tokens: 应始终保留为单个 token、永不被拆分的字符串列表。
        """
        # TODO: 实现
        self.vocab = vocab #id -> bytes
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()} #bytes -> id
        self.pair_to_merge = {pair: i for i, pair in enumerate(merges)} #pair -> index
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        从序列化到磁盘的 vocab 和 merges 文件构造 Tokenizer。

        参数:
            vocab_filepath:  序列化的 vocab 字典文件路径。
            merges_filepath: 序列化的 merges 列表文件路径。
            special_tokens:  可选的特殊 token 字符串列表。

        返回值:
            Tokenizer 实例。
        """
        # TODO: 实现
        # vocab: JSON 格式，key 是字符串形式的 int，value 是 bytes 的 base64 或 list[int]
        with open(vocab_filepath) as f:
            raw = json.load(f)
        # 假设存的格式是 {"0": [104], "1": [101], ...}
        vocab = {int(k): bytes(v) for k, v in raw.items()}
        
        # merges: 每行 "token1_bytes token2_bytes"
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ", 1)
                    merges.append((a.encode('utf-8'), b.encode('utf-8')))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将字符串编码为 token ID 列表。

        步骤（第 2.6.1 节）:
          1. 预分词文本（使用与训练时相同的正则表达式）。
          2. 按创建顺序对每个预分词单元应用 BPE 合并。
          3. 在词汇表中查找每个 token 对应的 ID。
        """
        # TODO: 实现
        #预分词
        full_text = text
        special_tokens = self.special_tokens
        # 1. 构造匹配所有特殊 token 的模式，用 | 连接
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(regex.escape(t) for t in special_tokens)
        # 2. 按特殊 token 切分，括号保留分隔符本身
        parts = []
        if special_tokens:
            parts = regex.split(f"({special_pattern})", full_text)
        else:
            parts = [full_text]
        # "Héllò hôw <|endoftext|> are ü?" → ["Héllò hôw ", "<|endoftext|>", " are ü?"]
        # 3. 对每个部分分别处理
        words = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for part in parts:
            if part in special_tokens:
                # 特殊 token: 直接加入（整个作为 bytes 序列）
                words.append([part.encode('utf-8')])
            else:
                # 普通文本: 正常预分词
                for token in regex.findall(PAT, part):
                    words.append([bytes([b]) for b in token.encode('utf-8')])
        
        #对每个预分词单元应用 BPE 合并
        for i in range(len(words)):
            word = words[i]
            while True:
                pairs = [(word[j], word[j + 1]) for j in range(len(word) - 1)]
                #找rank最低的pair
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
        #在词汇表中查找每个 token 对应的 ID
        ids = []
        for word in words:
            for token in word:
                token_id = self.token_to_id.get(token)
                if token_id is not None:
                    ids.append(token_id)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        惰性编码一个可迭代的字符串序列（例如文件句柄）。

        内存高效：逐个产出 token ID，无需将整个输入加载到内存中。
        确保 token 不会跨越数据块边界。
        """
        # TODO: 实现
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码回 Unicode 字符串。

        在词汇表中查找每个 ID 对应的字节序列，拼接后以 UTF-8 解码。
        无效字节将被替换为 Unicode 替换字符 U+FFFD。
        """
        # TODO: 实现
        tokens = [self.id_to_token[id] for id in ids]
        return b''.join(tokens).decode('utf-8', errors='replace')
