"""台词定位：在 SRT 正文中检索短句，定位到字幕输入序号与时间区间。

匹配模型：

1. 每条字幕正文先做 Unicode ``casefold`` 折叠，再剔除空白（``str.isspace``）
   与 Unicode 标点（general category 以 ``P`` 开头）字符，得到归一化字符
   流；同时保留每个归一化字符到「所在字幕 + 原正文偏移」的映射（casefold
   可能把一个字符展开成多个，如 ``ß`` → ``ss``，展开字符都映射回同一原文
   偏移）。待查短句做同样的归一化；剔除后为空的短句是参数错误。
2. 按**输入顺序**把相邻字幕的归一化正文拼接起来，但仅当「前条结束到后条
   开始」的间隔不超过阈值时才允许跨条匹配：间隔按
   ``max(0, 后条开始 − 前条结束)`` 计，时间重叠（间隔为负）视为 0。不可
   跨条的边界把整部字幕切成若干极大可拼接段，命中绝不跨越段边界。
3. 在每段的拼接字符流上用子串查找找出**全部命中（含重叠命中）**，再借助
   映射把每次命中拆回各条字幕的**实际命中范围**：每条字幕的原文片段只覆盖
   该条被命中的首末字符之间的原文（中间被忽略的空白 / 标点自然包含在
   内），不带入相邻文字。命中的时间区间取首条命中字幕的开始到末条命中
   字幕的结束。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .srt import Cue


class LocateError(ValueError):
    """待查短句剔除空白与标点后为空。"""


@dataclass(frozen=True)
class HitSegment:
    """一次命中在某条字幕内的实际命中范围（原正文字符偏移，半开区间）。"""

    cue_order: int
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class LocateHit:
    """一次命中：起止字幕输入序号（从 1 开始）与逐条命中片段。"""

    hit_order: int
    cue_order_from: int
    cue_order_to: int
    segments: tuple[HitSegment, ...]


def _is_ignored(ch: str) -> bool:
    """归一化时被忽略的字符：Unicode 空白或标点（P* 类别）。"""
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """对 ``text`` 做 casefold 并剔除空白 / 标点，返回归一化串与偏移映射。

    ``offsets[i]`` 是第 i 个归一化字符在原文中的字符偏移；casefold 展开
    出的多个字符映射回同一原文偏移。
    """
    chars: list[str] = []
    offsets: list[int] = []
    for pos, ch in enumerate(text):
        for folded in ch.casefold():
            if _is_ignored(folded):
                continue
            chars.append(folded)
            offsets.append(pos)
    return "".join(chars), offsets


def normalize_query(query: str) -> str:
    """把待查短句归一化为比较形式；剔除空白 / 标点后为空时报参数错误。"""
    normalized, _ = normalize_with_mapping(query)
    if not normalized:
        raise LocateError(
            "待查短句剔除空白与 Unicode 标点后为空，请填写实际要查找的台词。"
        )
    return normalized


def _bridgeable(prev: Cue, nxt: Cue, max_gap_ms: int) -> bool:
    """相邻两条字幕是否允许跨条匹配：间隔（重叠按 0 计）不超过阈值。"""
    return max(nxt.start - prev.end, 0) <= max_gap_ms


def _find_in_segment(
    normalized: list[tuple[str, list[int]]],
    lo: int,
    hi: int,
    needle: str,
) -> list[tuple[int, int, tuple[HitSegment, ...]]]:
    """在可跨条段 ``[lo, hi)`` 的拼接字符流中找出全部命中（含重叠命中）。

    返回 ``(起始输入序号, 结束输入序号, 逐条命中片段)`` 列表，按命中在字符
    流中的出现位置升序。
    """
    parts: list[str] = []
    char_map: list[tuple[int, int]] = []  # 归一化字符 -> (字幕下标, 原正文偏移)
    for k in range(lo, hi):
        norm, offsets = normalized[k]
        parts.append(norm)
        char_map.extend((k, off) for off in offsets)
    haystack = "".join(parts)

    found: list[tuple[int, int, tuple[HitSegment, ...]]] = []
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos < 0:
            break
        end = pos + len(needle)
        # 把命中区间 [pos, end) 按字幕拆成连续段，逐段映射回原正文偏移：
        # 片段覆盖该条被命中的首末字符之间的原文（含中间被忽略的字符）
        segments: list[HitSegment] = []
        run_cue, run_from = char_map[pos]
        prev_off = run_from
        for p in range(pos + 1, end):
            cue_idx, off = char_map[p]
            if cue_idx != run_cue:
                segments.append(HitSegment(run_cue + 1, run_from, prev_off + 1))
                run_cue, run_from = cue_idx, off
            prev_off = off
        segments.append(HitSegment(run_cue + 1, run_from, prev_off + 1))
        found.append(
            (segments[0].cue_order, segments[-1].cue_order, tuple(segments))
        )
        # 允许重叠命中：下一次查找只前进一个归一化字符
        start = pos + 1
    return found


def find_line_hits(
    cues: list[Cue], needle: str, max_gap_ms: int
) -> list[LocateHit]:
    """在 ``cues``（输入顺序）上查找归一化短句 ``needle`` 的全部命中。

    调用前 ``needle`` 应已通过 :func:`normalize_query` 归一化（非空）。
    命中按字符流中的出现顺序（字幕输入顺序 + 条内偏移）排列，含重叠命中。
    """
    if not needle:
        raise LocateError(
            "待查短句剔除空白与 Unicode 标点后为空，请填写实际要查找的台词。"
        )

    # 每条字幕的归一化正文与「归一化字符 -> 原正文偏移」映射
    normalized: list[tuple[str, list[int]]] = [
        normalize_with_mapping(cue.text) for cue in cues
    ]

    raw_hits: list[tuple[int, int, tuple[HitSegment, ...]]] = []
    n = len(cues)
    lo = 0
    while lo < n:
        # 极大可跨条段 [lo, hi)：段内相邻字幕的间隔均不超过阈值
        hi = lo + 1
        while hi < n and _bridgeable(cues[hi - 1], cues[hi], max_gap_ms):
            hi += 1
        raw_hits.extend(_find_in_segment(normalized, lo, hi, needle))
        lo = hi

    # 各段按输入顺序处理、段内按出现位置升序，整体即命中顺序
    return [
        LocateHit(
            hit_order=order,
            cue_order_from=cue_from,
            cue_order_to=cue_to,
            segments=segments,
        )
        for order, (cue_from, cue_to, segments) in enumerate(raw_hits, start=1)
    ]
