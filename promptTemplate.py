# Import MessagesPlaceholder for including multiple messages in a template
from langchain_core.prompts import MessagesPlaceholder
# Import HumanMessage for creating message objects with specific roles
from langchain_core.messages import HumanMessage
# Import the ChatPromptTemplate class from langchain_core.prompts module
from langchain_core.prompts import ChatPromptTemplate

from utils.inference_llm import inference_llm
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# Create a ChatPromptTemplate with a system message and a placeholder for multiple messages
# The system message sets the behavior for the assistant
# MessagesPlaceholder allows for inserting multiple messages at once into the template
prompt = ChatPromptTemplate.from_messages([
("system", "You are a helpful assistant"),
MessagesPlaceholder("msgs")  # This will be replaced with one or more messages
])

# Create an input dictionary where the key matches the MessagesPlaceholder name
# The value is a list of message objects that will replace the placeholder
# Here we're adding a single HumanMessage asking about the day after Tuesday
input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}

# Format the chat template with our input dictionary
# This replaces the MessagesPlaceholder with the HumanMessage in our input
# The result will be a formatted chat structure with a system message and our human message
prompt.invoke(input_)

llama='meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
llma_llm = WatsonxLLM(inference_llm(
    model_id = llama
))

chain = prompt | llma_llm
response = chain.invoke(input = input_)
print(response)