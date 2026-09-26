"""台词定位：在 SRT 正文中定位一句短台词，支持间隔阈值内的跨条匹配。

用户给出待查短句与允许跨条匹配的最大间隔毫秒数：

1. 按输入顺序把字幕切成若干**连续段（run）**：相邻两条字幕之间，仅当
   「后条开始 − 前条结束」不超过阈值时才允许同段跨条；时间重叠（含倒序）
   时间隔按零计，即 ``max(0, start_next − end_prev)``。间隔超过阈值的
   相邻字幕之间断开，跨断口的匹配不可能命中；
2. 段内按输入顺序直接拼接相邻字幕正文（不插入任何分隔字符），对短句与
   拼接正文做同样的归一化：逐字符判断——Unicode 空白（``str.isspace``）
   与 Unicode 标点类别（``unicodedata.category`` 以 ``P`` 开头）一律剔除，
   其余字符做 ``casefold``；``casefold`` 展开出多个字符时每个折叠字符都
   映射回同一个原偏移；
3. 保留归一化字符到「拼接正文偏移 → 原字幕及该条正文内偏移」的映射，
   在归一化串上找出短句的**全部子串命中（含相互重叠的命中）**；
4. 每处命中按实际涉及的字幕拆成若干片段：片段原文按该条内的实际命中
   偏移范围截取，命中范围内部的空白 / 标点会保留，但不带入条内命中范围
   之外的相邻文字；命中时间区间从首片段所在条的开始到末片段所在条的结束。
"""
from __future__ import annotations

import unicodedata
from bisect import bisect_right
from dataclasses import dataclass

from .srt import Cue


class LocateError(ValueError):
    """定位参数不合法（如归一化后短句为空）。"""


@dataclass(frozen=True)
class Normalized:
    """归一化结果：``text`` 为折叠后的比较串，``offsets`` 为每个折叠字符
    在原串中的字符偏移（casefold 一对多展开时多个折叠字符共用同一偏移）。"""

    text: str
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class HitSegment:
    """一处命中在单条字幕内的实际片段。"""

    cue_order: int
    index: str
    start: int
    end: int
    snippet: str
    match_length: int


@dataclass(frozen=True)
class LineHit:
    """一处命中：可能落在单条内，也可能跨多条。"""

    order: int
    cue_order_from: int
    cue_order_to: int
    start: int
    end: int
    segments: tuple[HitSegment, ...]

    @property
    def cross_cue(self) -> bool:
        return self.cue_order_to > self.cue_order_from

    @property
    def snippet(self) -> str:
        """各条片段按顺序直接拼接（不补任何相邻文字）。"""
        return "".join(segment.snippet for segment in self.segments)


@dataclass(frozen=True)
class _Run:
    """一组间隔都不超过阈值、允许跨条匹配的连续字幕。"""

    cue_from: int
    cue_to: int
    texts: tuple[str, ...]
    normalized: Normalized


def _is_skipped(ch: str) -> bool:
    """空白字符或 Unicode 标点类别（P*）字符在归一化时剔除。"""
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def normalize(text: str) -> Normalized:
    """剔除 Unicode 空白与标点后做 casefold，并保留到原偏移的映射。"""
    chars: list[str] = []
    offsets: list[int] = []
    for pos, ch in enumerate(text):
        if _is_skipped(ch):
            continue
        for folded in ch.casefold():
            chars.append(folded)
            offsets.append(pos)
    return Normalized("".join(chars), tuple(offsets))


def gap_ms(previous: Cue, current: Cue) -> int:
    """前条结束到后条开始的间隔毫秒数；时间重叠（含倒序）视为间隔零。"""
    return max(0, current.start - previous.end)


def _build_runs(cues: list[Cue], max_gap_ms: int) -> list[_Run]:
    """按相邻间隔阈值把字幕切成允许跨条匹配的连续段。"""
    runs: list[_Run] = []
    i = 0
    n = len(cues)
    while i < n:
        j = i
        texts = [cues[i].text]
        while j + 1 < n and gap_ms(cues[j], cues[j + 1]) <= max_gap_ms:
            j += 1
            texts.append(cues[j].text)
        runs.append(
            _Run(
                cue_from=i,
                cue_to=j,
                texts=tuple(texts),
                normalized=normalize("".join(texts)),
            )
        )
        i = j + 1
    return runs


def locate_lines(cues: list[Cue], query: str, max_gap_ms: int) -> list[LineHit]:
    """在 ``cues`` 中找出短句的全部命中（含重叠命中），按文档出现顺序排列。"""
    needle = normalize(query)
    if not needle.text:
        raise LocateError(
            "待查短句在忽略空白与 Unicode 标点后为空，请填写实际要定位的文字。"
        )

    hits: list[LineHit] = []
    order = 0
    for run in _build_runs(cues, max_gap_ms):
        haystack = run.normalized.text
        if len(haystack) < len(needle.text):
            continue

        # 各条正文在拼接串中的起点偏移，用于把拼接偏移还原为（条, 条内偏移）
        bases: list[int] = []
        base = 0
        for text in run.texts:
            bases.append(base)
            base += len(text)

        def cue_at(global_offset: int) -> int:
            return bisect_right(bases, global_offset) - 1

        pos = haystack.find(needle.text)
        while pos != -1:
            end = pos + len(needle.text)
            global_offsets = run.normalized.offsets

            # 把命中的归一化位置按所属字幕分组，逐条截取实际命中范围
            segments: list[HitSegment] = []
            k = pos
            while k < end:
                local_index = cue_at(global_offsets[k])
                k2 = k + 1
                while k2 < end and cue_at(global_offsets[k2]) == local_index:
                    k2 += 1

                lo = global_offsets[k] - bases[local_index]
                hi = global_offsets[k2 - 1] - bases[local_index] + 1
                cue = cues[run.cue_from + local_index]
                segments.append(
                    HitSegment(
                        cue_order=run.cue_from + local_index + 1,
                        index=cue.index,
                        start=cue.start,
                        end=cue.end,
                        snippet=run.texts[local_index][lo:hi],
                        match_length=k2 - k,
                    )
                )
                k = k2

            order += 1
            hits.append(
                LineHit(
                    order=order,
                    cue_order_from=segments[0].cue_order,
                    cue_order_to=segments[-1].cue_order,
                    start=segments[0].start,
                    end=segments[-1].end,
                    segments=tuple(segments),
                )
            )
            # 起点只前进一位：保留相互重叠的命中
            pos = haystack.find(needle.text, pos + 1)

    return hits
