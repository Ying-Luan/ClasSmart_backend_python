from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 定义Agent
model_client = OpenAIChatCompletionClient(
    model='gpt-4o-mini',
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('OPENAI_API_KEY')
)

agent = AssistantAgent(
    name='imageSelectAgent',
    model_client=model_client,
    system_message="""You are a helpful assistant for identifying pictures, 
    the default image content types are battery, bottle, can, China, fruit, smoke, vegetable, brick, 
    I will give you the picture, and the corresponding label, 
    your answer will strictly respect the following rules, 
    when you think the picture meets the given label, 
    directly answer 1, 
    when you think the picture does not meet the given label,
    directly answer 0,
    do not answer other decorative content
    """,
    description="""A helpful assistant agent for identifying pictures, 
    the default image content types are battery, bottle, can, China, fruit, smoke, vegetable, brick, 
    I will give you the picture, and the corresponding label, 
    your answer will strictly respect the following rules, 
    when you think the picture meets the given label, 
    directly answer 1, 
    when you think the picture does not meet the given label,
    directly answer 0,
    do not answer other decorative content
    """
)