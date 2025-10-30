import json
import os
import queue
import re
import time
from threading import Thread

import httpx
from modelscope import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from transformers import TextIteratorStreamer, TextStreamer


class Qwen:
    """本地部署的Qwen大语言模型推理类。"""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        """
        初始化模型和分词器。

        参数:
            model_name (str): Hugging Face或ModelScope上的模型名称。
        """
        # 从预训练模型加载因果语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",  # 自动选择合适的torch数据类型
            device_map="auto",  # 自动将模型分配到可用设备（如GPU）
        )
        # 从预训练模型加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def infer(self, user_input, user_messages, chat_mode):
        """
        执行非流式推理。

        参数:
            user_input (str): 用户当前输入。
            user_messages (list): 包含对话历史的列表。
            chat_mode (str): 对话模式（单轮或互动）。

        返回:
            tuple: (模型回复, 更新后的对话历史)
        """
        # 根据对话模式和历史设置系统提示(prompt)
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]["content"] = (
                    "你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”等短句作为回复的开头。"
                )
            else:
                with open("src/prompt.txt", "r") as f:
                    user_messages[0]["content"] = f.read()
        user_messages.append({"role": "user", "content": user_input})
        print(user_messages)

        # 应用聊天模板，生成模型输入文本
        text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成回复
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,  # 最大生成token数
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的ID为文本
        chat_response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        user_messages.append({"role": "assistant", "content": chat_response})

        # 保持对话历史不超过10轮
        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f"[Qwen] {chat_response}")
        return chat_response, user_messages

    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        """
        执行流式推理，并将生成的句子放入队列。

        参数:
            user_input (str): 用户当前输入。
            user_messages (list): 对话历史。
            llm_queue (queue.Queue): 用于存放生成句子的队列。
            chunk_size (int): 句子分块的最小长度。
            chat_mode (str): 对话模式。

        返回:
            tuple: (完整回复, 更新后的对话历史, 每句话的时间成本)
        """
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # 设置系统提示
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]["content"] = (
                    "你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”、“当然可以”等短句作为回复的开头。"
                )
            else:
                with open("src/prompt.txt", "r") as f:
                    user_messages[0]["content"] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({"role": "user", "content": user_input})

        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 设置文本流式生成器
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"}
        )
        # 在新线程中运行模型生成，以实现非阻塞
        thread = Thread(
            target=self.model.generate,
            kwargs={**model_inputs, "streamer": streamer, "max_new_tokens": 512},
        )
        thread.start()

        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        # 使用正则表达式按标点符号分割句子
        sentence_split_pattern = re.compile(r"(?<=[,;.!?，；：。:！？》、”])")
        fp_flag = True  # 首句标志
        print("[LLM] Start LLM streaming...")
        # 遍历流式生成器产出的每个token
        for chunk in streamer:
            chat_response_chunk = chunk
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            # 按标点分割句子
            sentences = sentence_split_pattern.split(buffer)

            if not sentences:
                continue

            # 处理除最后一个（可能不完整）之外的所有句子
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                # 如果是首句或累积的句子长度达到阈值，则放入队列
                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time() - start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False

            # 更新缓冲区为最后一个（可能不完整的）句子
            buffer = sentences[-1].strip()

        # 处理剩余的缓冲区内容
        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        # 放入None作为流结束的信号
        llm_queue.put(None)

        # 更新对话历史
        user_messages.append({"role": "assistant", "content": chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f"[LLM] Response: {chat_response}\n")

        return chat_response, user_messages, time_cost


class Qwen_API:
    """通过API调用Qwen大语言模型的推理类（兼容OpenAI API格式）。"""

    def __init__(
        self, api_key=None, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        """
        初始化OpenAI客户端。

        参数:
            api_key (str, optional): API密钥。如果为None，则从环境变量读取。
            base_url (str): API的基地址。
        """
        api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        # 创建一个不使用代理的HTTP客户端以避免代理相关的兼容性问题
        # 通过设置trust_env=False来忽略系统代理环境变量
        http_client = httpx.Client(trust_env=False)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    def infer(self, user_input, user_messages, chat_mode):
        """
        执行非流式API推理。
        """
        # 设置系统提示
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]["content"] = (
                    "你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”等短句作为回复的开头。"
                )
            else:
                with open("src/prompt.txt", "r") as f:
                    user_messages[0]["content"] = f.read()
        user_messages.append({"role": "user", "content": user_input})
        print(user_messages)

        # 调用API
        completion = self.client.chat.completions.create(
            model="qwen-turbo", messages=user_messages
        )
        print(completion)
        chat_response = completion.choices[0].message.content
        user_messages.append({"role": "assistant", "content": chat_response})

        # 管理对话历史长度
        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f"[Qwen API] {chat_response}")
        return chat_response, user_messages

    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        """
        执行流式API推理。
        """
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # 设置系统提示
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]["content"] = (
                    "你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”、“当然可以”等短句作为回复的开头。"
                )
            else:
                with open("src/prompt.txt", "r") as f:
                    user_messages[0]["content"] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({"role": "user", "content": user_input})

        # 调用流式API
        completion = self.client.chat.completions.create(
            model="qwen-turbo", messages=user_messages, stream=True
        )

        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r"(?<=[,;.!?，；：。:！？》、”])")
        fp_flag = True  # 首句标志
        print("[LLM] Start LLM streaming...")
        # 处理流式返回的每个数据块
        for chunk in completion:
            if chunk.choices[0].delta.content is None:
                continue
            chat_response_chunk = chunk.choices[0].delta.content
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)

            if not sentences:
                continue

            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time() - start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False

            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        # 流结束标志
        llm_queue.put(None)

        # 更新对话历史
        user_messages.append({"role": "assistant", "content": chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)

        print(f"[LLM] Response: {chat_response}\n")

        return chat_response, user_messages, time_cost


if __name__ == "__main__":
    # 本地测试代码
    start_time = time.time()
    qwen = Qwen()  # 实例化本地模型
    print(f"Cost {time.time() - start_time} secs")
    start_time = time.time()
    # 测试流式推理
    qwen.infer_stream(
        "讲一个长点的故事",
        [{"role": "system", "content": None}],
        queue.Queue(),
        10,
        "单轮对话 (一次性回答问题)",
    )
    print(f"Cost {time.time() - start_time} secs")
