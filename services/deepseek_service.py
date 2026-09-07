import re
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import config.config as config
from utils.logger import logger
from utils.decorators import retry_on_exception

class DeepSeekService:
    def __init__(self):
        self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    @staticmethod
    def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
        """将对话历史渲染为提示词文本；无历史返回空串"""
        if not history:
            return ""
        lines = []
        for m in history:
            role_label = "用户" if m.get("role") == "user" else "助手"
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def fallback_rule_based(self, query: str) -> Dict[str, Any]:
        """降级兜底规则（电商客服关键词表）。检查顺序敏感：投诉/订单类须先于售后类判断"""
        escalation_keywords = ["投诉", "转人工", "人工客服", "客服电话", "12315", "差评", "举报", "赔偿", "太差了", "垃圾", "态度"]
        if any(kw in query for kw in escalation_keywords):
            return {"intent": "escalation", "description": "规则引擎匹配（投诉/要求人工）"}

        order_keywords = ["订单", "物流", "快递", "发货", "到货", "单号", "签收", "退款进度"]
        if any(kw in query for kw in order_keywords):
            return {"intent": "order_status", "description": "规则引擎匹配（订单/物流查询）"}

        after_sales_keywords = ["退货", "换货", "保修", "维修", "质保", "发票", "运费", "售后", "退差价", "无理由"]
        if any(kw in query for kw in after_sales_keywords):
            return {"intent": "after_sales", "description": "规则引擎匹配（售后政策）"}

        promotion_keywords = ["优惠", "满减", "折扣", "活动", "赠品", "优惠券", "以旧换新", "学生价", "秒杀", "价保", "积分"]
        if any(kw in query for kw in promotion_keywords):
            return {"intent": "promotion", "description": "规则引擎匹配（促销活动）"}

        comparison_keywords = ["对比", "哪个好", "区别", "相比", "哪个性价比", "vs"]
        if any(kw in query for kw in comparison_keywords):
            return {"intent": "comparison", "description": "规则引擎匹配（商品对比）"}

        pre_sales_keywords = ["参数", "配置", "价格", "多少钱", "续航", "电池", "屏幕", "摄像头", "快充", "防水", "颜色", "内存", "处理器", "现货"]
        if any(kw in query for kw in pre_sales_keywords):
            return {"intent": "pre_sales", "description": "规则引擎匹配（售前咨询）"}

        chitchat_keywords = ["你好", "在吗", "谢谢", "再见", "你是谁", "哈哈", "早上好", "晚上好"]
        if any(kw in query for kw in chitchat_keywords):
            return {"intent": "chitchat", "description": "规则引擎匹配（寒暄闲聊）"}

        return {"intent": "other", "description": "规则未命中，降级为默认通用类"}

    @retry_on_exception()
    def recognize_intent(self, query: str) -> Dict[str, Any]:
        """意图识别分类服务"""
        prompt = config.INTENT_PROMPT_TEMPLATE.format(query=query)
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严格的分类器，只输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            intent_info = json.loads(json_match.group())
        else:
            raise ValueError("未从 DeepSeek 返回中提取到有效的 JSON Structure")

        if intent_info.get("confidence", 0) < 0.7:
            return self.fallback_rule_based(query)
        return intent_info

    @retry_on_exception()
    def rewrite_queries(self, original_query: str, num_rewrites: int = 5,
                        history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """多路问句改写生成；history 为多轮对话历史（用于指代消解），无历史时行为与单轮一致"""
        history_text = self._format_history(history)
        if history_text:
            prompt = f"""你是一个查询改写助手。以下是用户与系统的多轮对话历史。请根据对话历史，将用户的当前问题改写为{num_rewrites}个语义相同但表述不同的独立问句，必须补全指代不明的代词、人名、对象等，使每个问句脱离上下文也能独立理解，以便更好地检索信息。直接输出改写后的问句，每行一个，不要有多余内容。
对话历史：
{history_text}

用户当前问题：{original_query}
改写："""
        else:
            prompt = f"""你是一个查询改写助手。请根据用户原问题，生成{num_rewrites}个语义相同但表述不同的问句，以便更好地检索信息。直接输出改写后的问句，每行一个，不要有多余内容。
原问题：{original_query}
改写："""
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是查询改写专家，只输出改写问句，每行一个。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        content = response.choices[0].message.content.strip()
        rewrites = [line.strip() for line in content.split('\n') if line.strip()]
        if not rewrites:
            # 空结果视为失败：抛异常触发 retry_on_exception 重试（LLM 偶发返回空，实测约 1/14 概率）
            raise ValueError("改写返回空结果")

        unique = []
        seen = set()
        for q in [original_query] + rewrites:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]],
                        history: Optional[List[Dict[str, str]]] = None) -> str:
        """基于检索片段生成客服回答；history 为多轮对话历史（用于指代理解）"""
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            context_text += f"[文档{i}] {doc['chunk_text']}\n\n"

        history_text = self._format_history(history)
        if history_text:
            prompt = f"""你是星曜官方商城的资深客服助手，请根据以下提供的文档片段回答用户问题。

【回答要求】
1. 先共情或直接给出结论，语气口语化、友好、简洁；
2. 涉及退换货、保修、运费等政策时严格按文档原文表述，不推测、不扩大承诺；
3. 文档中没有的信息，请直接回答"很抱歉，关于这个问题我暂时没有查到相关信息，建议您联系人工客服（工作时间 9:00-21:00）进一步确认"，严禁编造；
4. 结合对话历史理解用户当前问题中的指代关系。

对话历史：
{history_text}

用户当前问题：{query}

提供的文档片段：
{context_text}

回答："""
        else:
            prompt = f"""你是星曜官方商城的资深客服助手，请根据以下提供的文档片段回答用户问题。

【回答要求】
1. 先共情或直接给出结论，语气口语化、友好、简洁；
2. 涉及退换货、保修、运费等政策时严格按文档原文表述，不推测、不扩大承诺；
3. 文档中没有的信息，请直接回答"很抱歉，关于这个问题我暂时没有查到相关信息，建议您联系人工客服（工作时间 9:00-21:00）进一步确认"，严禁编造。

用户问题：{query}

提供的文档片段：
{context_text}

回答："""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": "你是星曜官方商城的资深客服助手，只依据提供的文档片段回答，绝不编造。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek 答案生成失败: {e}", exc_info=True)
            return "抱歉，生成回答时出现错误，请稍后重试。"

    def _chat(self, system: str, prompt: str, temperature: float = 0.3, max_tokens: int = 2048,
              model: str = "deepseek-v4-pro") -> str:
        """共享的 OpenAI 调用；失败时返回降级话术（不抛异常，保证分支链路不中断）"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek 调用失败（{system[:20]}...）: {e}", exc_info=True)
            return "抱歉，当前服务繁忙，请稍后重试或联系人工客服。"

    @retry_on_exception()
    def extract_order_info(self, query: str) -> Dict[str, Any]:
        """从用户消息中抽取订单号（JSON 抽取，复用 recognize_intent 的解析模式）"""
        prompt = f"""你是电商客服系统的订单信息抽取器。从用户消息中抽取订单号（形如 YX20260815001 的字母数字编号）。
只输出一个 JSON 对象：{{"order_id": "抽取到的订单号，没有则为空字符串", "confidence": 0.0到1.0的小数, "note": "一句话说明"}}
不要输出 JSON 以外的任何内容。
用户消息：{query}"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是严格的订单信息抽取器，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("未从 DeepSeek 返回中提取到有效的订单 JSON")
        return json.loads(json_match.group())

    def generate_order_answer(self, query: str, order: Dict[str, Any],
                              history: Optional[List[Dict[str, str]]] = None) -> str:
        """基于 mock 订单 JSON 生成订单查询回答，严禁编造物流信息"""
        history_text = self._format_history(history)
        history_block = f"对话历史：\n{history_text}\n\n" if history_text else ""
        prompt = f"""你是星曜官方商城的客服助手。只能依据给定的订单信息回答，严禁编造物流单号、时间或状态。

【回答要求】
1. 先礼貌称呼并直接给出结论；
2. 说明订单当前状态、最新一条物流动态、预计送达时间；
3. 若订单处于退款中/退款成功，说明退款进度与到账时效；
4. 用户询问订单信息以外的问题（如发票、改地址）时，明确说明订单信息中查不到，并引导转人工；
5. 时间必须按订单数据原文引用（如“今日 09:15”），严禁自行换算成今天/昨天/某日期，严禁推断数据中没有的信息；
6. 口语化、简短（3-5 句）。

{history_block}订单信息：
{order}

用户问题：{query}
回答："""
        return self._chat("你是星曜官方商城的客服助手，只依据订单信息回答，绝不编造。", prompt,
                          temperature=0.2, max_tokens=2048)

    def generate_escalation_reply(self, query: str,
                                  history: Optional[List[Dict[str, str]]] = None) -> str:
        """投诉/转人工安抚话术：共情致歉 → 承诺安排人工 → 告知服务时间"""
        history_text = self._format_history(history)
        history_block = f"对话历史：\n{history_text}\n\n" if history_text else ""
        prompt = f"""你是星曜官方商城的客服主管。用户表达了不满或要求人工处理，请回复安抚话术。

【回复要求】
1. 先共情并真诚致歉；
2. 表示已记录问题，会立即安排人工客服跟进；
3. 告知人工客服工作时间为每天 9:00-21:00，预计 5 分钟内响应；
4. 安抚语气，不超过 3-4 句，结尾保留转人工承诺。

{history_block}用户消息：{query}
回复："""
        return self._chat("你是星曜官方商城的客服主管，善于安抚客户情绪。", prompt,
                          temperature=0.5, max_tokens=512)

    def generate_chitchat_reply(self, query: str,
                                history: Optional[List[Dict[str, str]]] = None) -> str:
        """寒暄闲聊简短回复，并引导到业务话题"""
        history_text = self._format_history(history)
        history_block = f"对话历史：\n{history_text}\n\n" if history_text else ""
        prompt = f"""你是星曜商城客服小星。用户发来问候或闲聊，请简短友好回应。

【回复要求】
1. 1-2 句即可，自我介绍是星曜商城客服小星；
2. 主动引导用户可以咨询：商品选购、订单物流查询、售后服务三类问题。

{history_block}用户消息：{query}
回复："""
        return self._chat("你是星曜商城客服小星，热情友好。", prompt,
                          temperature=0.7, max_tokens=256)