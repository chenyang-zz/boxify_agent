"""长期记忆图谱受控词表。"""

ENTITY_TYPES: dict[str, str] = {
    "生命体": "稳定指向的生命体个体，例如用户、家人、朋友、宠物。",
    "组织": "公司、学校、团队、社群等组织主体。",
    "群体": "边界相对稳定的一组人。",
    "角色职业": "社会角色、功能身份或职业身份。",
    "地点设施": "地理位置或功能性空间。",
    "物品设备": "可持有或使用的具体物品、设备、工具。",
    "软件平台": "软件、应用、网站或数字服务。",
    "识别联系信息": "账号、用户名、邮箱、手机号等标识或联系信息。",
    "文档媒体": "文章、报告、图片、音视频等内容载体。",
    "知识能力": "知识主题、技能、学科或语言。",
    "偏好习惯": "稳定偏好、重复习惯或长期行为倾向。",
    "具体目标": "明确、可验证、可长期追踪的目标。",
    "称呼别名": "用于指代或称呼实体的名字。",
}

PREDICATES: dict[str, str] = {
    "别名属于": "别名指向对应规范实体。",
    "属于类型": "实体属于某种类别、身份、角色或归属对象。",
    "位于": "实体位于某地点、场所或空间位置。",
    "前往": "主体前往某个地点、组织或活动对象。",
    "组成部分": "实体是另一实体的组成部分。",
    "拥有": "主体拥有、持有或配有某对象、账号、联系方式。",
    "使用": "主体使用某工具、平台、语言或资源。",
    "创建了": "主体创建、撰写或生产某对象。",
    "了解": "主体了解、学习或持续关注某知识主题。",
    "偏好": "主体对某对象具有稳定偏好、厌恶或目标倾向。",
    "负责": "主体负责某项工作、职责、事务或领域。",
    "沟通于": "两个实体之间发生沟通或交流。",
    "关联于": "存在明确、稳定且有记忆价值的弱联系。",
}

UNKNOWN_ENTITY_TYPE = "其他"
UNKNOWN_PREDICATE = "关联于"


def normalize_entity_type(value: str | None) -> str:
    """将实体类型规范到受控词表。"""
    label = (value or "").strip()
    return label if label in ENTITY_TYPES else UNKNOWN_ENTITY_TYPE


def normalize_predicate(value: str | None) -> str:
    """将关系谓词规范到受控词表。"""
    label = (value or "").strip()
    return label if label in PREDICATES else UNKNOWN_PREDICATE


def format_ontology_for_prompt() -> str:
    """渲染给 LLM prompt 使用的受控词表。"""
    entity_lines = [f"- {name}: {desc}" for name, desc in ENTITY_TYPES.items()]
    predicate_lines = [f"- {name}: {desc}" for name, desc in PREDICATES.items()]
    return (
        "实体类型只能选择：\n"
        + "\n".join(entity_lines)
        + "\n\n关系谓词只能选择：\n"
        + "\n".join(predicate_lines)
    )
