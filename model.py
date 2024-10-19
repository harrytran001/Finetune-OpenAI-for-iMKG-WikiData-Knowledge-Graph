from typing import TypedDict
from openai.types.chat import ChatCompletionMessageParam

class TrainData(TypedDict):
    messages: ChatCompletionMessageParam