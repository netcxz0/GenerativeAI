# Import the JsonOutputParser from langchain_core to convert LLM responses into structured JSON
#from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
# Import BaseModel and Field from langchain_core's pydantic_v1 module
#from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from utils.llm_utils import llm_model
#from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
#from langchain_ibm import WatsonxLLM

# Define your desired data structure.
""" class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# And a query intended to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."
 """
# Set up a parser + inject instructions into the prompt template.
# output_parser = JsonOutputParser(pydantic_object=Joke)
output_parser = CommaSeparatedListOutputParser()

# Get the formatting instructions for the output parser
# This generates guidance text that tells the LLM how to format its response
format_instructions = output_parser.get_format_instructions()

# Create a prompt template that includes:
# 1. Instructions for the LLM to answer the user's query
# 2. Format instructions to ensure the LLM returns properly structured data
# 3. The actual user query placeholder
""" prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],  # Dynamic variables that will be provided when invoking the chain
    partial_variables={"format_instructions": format_instructions},  # Static variables set once when creating the prompt
) """
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],  # This variable will be provided when the chain is invoked
    partial_variables={"format_instructions": format_instructions},  # This variable is set once when creating the prompt
)

llama='meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
llama_llm = llm_model(llama)
 
# Create a processing chain that:
# 1. Formats the prompt using the template
# 2. Sends the formatted prompt to the Llama LLM
# 3. Parses the LLM's response using the output parser to extract structured data
chain = prompt | llama_llm | output_parser

# Invoke the chain with a specific query about jokes
# This will:
# 1. Format the prompt with the joke query
# 2. Send it to Llama
# 3. Parse the response into the structure defined by your output parser
# 4. Return the structured result
print(chain.invoke({"subject": "ice cream flavors"}))