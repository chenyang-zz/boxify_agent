# 长期记忆图谱萃取提示词模板

EXTRACT_STATEMENTS_SYSTEM_PROMPT = """
将用户长期记忆拆成原子陈述，返回 JSON。
"""

EXTRACT_STATEMENTS_PROMPT = """
只返回形如 {{"statements":[{{"text":"..."}}]}} 的 JSON。
文本：{text}
"""

EXTRACT_TRIPLETS_SYSTEM_PROMPT = """
从原子陈述中抽取实体三元组，返回 JSON。
"""

EXTRACT_TRIPLETS_PROMPT = """
只返回形如 {{"triplets":[{{"head":{{"name":"...","type":"...","description":"..."}},"relation":"...","tail":{{"name":"...","type":"...","description":"..."}},"evidence":"..."}}]}} 的 JSON。
陈述：{statements}
"""
