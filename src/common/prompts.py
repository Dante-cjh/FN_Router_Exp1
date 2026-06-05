"""Prompt templates.

Two families of prompts:

1) ARG-style *advisor* prompts (used by step1 to build the rationale dataset).
   ARG asks the LLM to judge each news item from two complementary
   perspectives and to write a short rationale for each:
       - td : textual description  (writing style, wording, structure, tone)
       - cs : common sense         (factual plausibility, world knowledge)
   For each perspective the LLM returns a {real|fake} judgment + a rationale.
   These become td_pred / cs_pred / td_rationale / cs_rationale in ARG format.

2) A *direct* judge prompt (used by step2) where the LLM makes a single
   holistic real/fake decision for the "GPT-5.4 direct" diagnostic model.

All prompts force a strict JSON reply so parsing is robust. Edit freely to
match the exact wording in the ARG paper appendix if you want a 1:1 reproduction.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# English (GossipCop)
# --------------------------------------------------------------------------- #
EN_TD_SYSTEM = (
    "You are a professional fake-news detector. You analyze a news article "
    "purely from the perspective of its TEXTUAL DESCRIPTION: writing style, "
    "wording, structure, tone, emotional manipulation, and signs of "
    "sensationalism or fabrication in how the text is written."
)
EN_CS_SYSTEM = (
    "You are a professional fake-news detector. You analyze a news article "
    "from the perspective of COMMON SENSE and world knowledge: whether the "
    "described events, entities, and claims are factually plausible and "
    "internally consistent with what is generally known about the world."
)
EN_PERSPECTIVE_USER = (
    "Decide whether the following news is real or fake from your perspective, "
    "and explain your reasoning in 2-4 sentences.\n\n"
    "News: {content}\n\n"
    'Reply with ONLY a JSON object: {{"prediction": "real" | "fake", '
    '"rationale": "<your reasoning>"}}'
)

EN_DIRECT_SYSTEM = (
    "You are a professional fake-news detector. Given a news article, make a "
    "single holistic judgment about whether it is real or fake, using both the "
    "writing and your world knowledge."
)
EN_DIRECT_USER = (
    "Is the following news real or fake?\n\n"
    "News: {content}\n\n"
    'Reply with ONLY a JSON object: {{"prediction": "real" | "fake", '
    '"rationale": "<one short sentence>"}}'
)

# --------------------------------------------------------------------------- #
# Chinese (Weibo21)
# --------------------------------------------------------------------------- #
ZH_TD_SYSTEM = (
    "你是一名专业的虚假新闻鉴别专家。你只从【文本描述】的角度分析一条新闻："
    "包括写作风格、用词、结构、语气、情绪煽动，以及文字本身是否存在夸张或编造的迹象。"
)
ZH_CS_SYSTEM = (
    "你是一名专业的虚假新闻鉴别专家。你从【常识与世界知识】的角度分析一条新闻："
    "判断其中描述的事件、人物和说法是否在事实上合理、是否与公认的常识一致、内部是否自洽。"
)
ZH_PERSPECTIVE_USER = (
    "请从你的角度判断下面这条新闻是真实(real)还是虚假(fake)，并用2-4句话说明理由。\n\n"
    "新闻：{content}\n\n"
    '只输出一个 JSON 对象：{{"prediction": "real" 或 "fake", "rationale": "<你的理由>"}}'
)

ZH_DIRECT_SYSTEM = (
    "你是一名专业的虚假新闻鉴别专家。给定一条新闻，请综合文字表达与世界知识，"
    "给出它是真实还是虚假的整体判断。"
)
ZH_DIRECT_USER = (
    "下面这条新闻是真实(real)还是虚假(fake)？\n\n"
    "新闻：{content}\n\n"
    '只输出一个 JSON 对象：{{"prediction": "real" 或 "fake", "rationale": "<一句话理由>"}}'
)


def get_perspective_prompt(language: str, perspective: str, content: str):
    """Return (system, user) for an ARG advisor perspective.

    perspective: 'td' (textual description) or 'cs' (common sense)
    """
    language = language.lower()
    perspective = perspective.lower()
    if language == "en":
        system = EN_TD_SYSTEM if perspective == "td" else EN_CS_SYSTEM
        user = EN_PERSPECTIVE_USER.format(content=content)
    elif language == "zh":
        system = ZH_TD_SYSTEM if perspective == "td" else ZH_CS_SYSTEM
        user = ZH_PERSPECTIVE_USER.format(content=content)
    else:
        raise ValueError(f"Unsupported language: {language}")
    return system, user


def get_direct_prompt(language: str, content: str):
    """Return (system, user) for the holistic direct judge."""
    language = language.lower()
    if language == "en":
        return EN_DIRECT_SYSTEM, EN_DIRECT_USER.format(content=content)
    elif language == "zh":
        return ZH_DIRECT_SYSTEM, ZH_DIRECT_USER.format(content=content)
    raise ValueError(f"Unsupported language: {language}")
