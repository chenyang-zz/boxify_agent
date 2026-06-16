from app.domain.services.memory.ontology import format_ontology_for_prompt

EXTRACT_STATEMENTS_SYSTEM_PROMPT = """
将用户长期记忆拆成原子陈述，完成指代消解，并返回严格 JSON。
"""

EXTRACT_STATEMENTS_PROMPT = """
只返回形如 {{"statements":[{{"statement":"...","statement_type":"FACT","temporal_type":"STATIC","has_unsolved_reference":false,"importance":0.5,"confidence":0.8}}]}} 的 JSON。
要求：
- 每条 statement 只表达一个事实、偏好、观点或目标。
- 将“我/我的”统一改写为“用户”。
- 如果指代无法消解，has_unsolved_reference 置为 true。
- statement_type 只能是 FACT / OPINION / PREDICTION / SUGGESTION。
- temporal_type 只能是 STATIC / DYNAMIC / ATEMPORAL。
文本：{text}
"""

EXTRACT_TRIPLETS_SYSTEM_PROMPT = """
从原子陈述中抽取实体和实体关系三元组，返回严格 JSON。
"""

EXTRACT_TRIPLETS_PROMPT = (
    format_ontology_for_prompt()
    + """

只返回形如 {{"entities":[{{"entity_idx":1,"name":"用户","type":"生命体","description":"当前用户","importance":0.5,"confidence":0.8}}],"triplets":[{{"subject_id":1,"predicate":"偏好","object_id":2,"evidence":"用户喜欢周杰伦。","importance":0.5,"confidence":0.8}}]}} 的 JSON。
要求：
- entities[].type 必须来自实体类型词表；拿不准时填“其他”。
- triplets[].predicate 必须来自关系谓词词表；拿不准时填“关联于”。
- subject_id/object_id 必须引用 entities[].entity_idx。
- 不要输出事件节点或时间线。
陈述：{statements}
"""
)

PROFILE_SUMMARY_SYSTEM_PROMPT = "你是记忆图谱画像增强器，只返回严格 JSON。"

PROFILE_SUMMARY_PROMPT = (
    "请基于实体相关陈述，生成长期记忆画像。\n"
    "只返回 JSON："
    '{{"core_facts":["..."],"traits":["..."]}}\n'
    "要求：\n"
    "- core_facts 是稳定、可复用的事实，最多 8 条。\n"
    "- traits 是偏好、倾向、习惯或风格，最多 8 条。\n"
    "- 不要编造陈述中没有的信息。\n"
    "实体：{entity_name}（{entity_type}）\n"
    "陈述：{statements}"
)

REFLECT_SYSTEM_PROMPT = "你是记忆图谱反思器，只返回严格 JSON。"

REFLECT_PROMPT = """
基于给定的「用户记忆清单」，归纳 {min_insights}~{max_insights} 条高层洞察。
每条洞察是对一类稳定信息的概括性结论，不要简单复述单条事实。

只返回形如：
{{"insights":[{{"theme":"音乐偏好","content":"用户偏好华语流行音乐。","based_on":["周杰伦"],"importance":0.8,"confidence":0.9}}]}}

要求：
- theme 是简短中文主题。
- content 是一句可复用的用户理解，不超过 200 字。
- based_on 只能引用记忆清单中出现过的实体名称。
- importance/confidence 必须是 0 到 1 之间的小数。
- 记忆清单过少或无法归纳时返回 {{"insights":[]}}。

记忆清单：
{memory_block}
"""

COMMUNITY_SUMMARY_SYSTEM_PROMPT = "你是记忆图谱社区命名器，只返回严格 JSON。"

COMMUNITY_SUMMARY_PROMPT = """
以下是一组语义相关的用户记忆实体和社区内部关系。
请为这个社区生成简洁中文名称和一句摘要。

只返回形如：
{{"name":"音乐偏好","summary":"用户的音乐兴趣、歌手偏好和相关作品。"}}

要求：
- name 不超过 10 个中文字符。
- summary 不超过 80 个中文字符。
- 不要编造实体和关系中不存在的信息。

实体：
{members}

关系：
{relationships}
"""
