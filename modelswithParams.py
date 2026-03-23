def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from utils.inference_llm import inference_llm

# Define different parameter sets
parameters_creative = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.8,  # Higher temperature for more creative responses
}

parameters_precise = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.1,  # Lower temperature for more deterministic responses
}

# Define the model ID for ibm/granite-3-3-8b-instruct
granite='ibm/granite-3-3-8b-instruct'

# Define the model ID for llama-4-maverick-17b-128e-instruct-fp8
llama='meta-llama/llama-4-maverick-17b-128e-instruct-fp8'

# Create two model instances with different parameters for Granite model
granite_creative = inference_llm(
    model_id=granite,
    params=parameters_creative
)

granite_precise = inference_llm(
    model_id=granite,
    params=parameters_precise
)

# Create two model instances with different parameters for Llama model
llama_creative = inference_llm(
    model_id=llama,
    params=parameters_creative
)

llama_precise = inference_llm(
    model_id=llama,
    params=parameters_precise
)


# Wrap them for LangChain for both models
granite_llm_creative = WatsonxLLM(model=granite_creative)
granite_llm_precise = WatsonxLLM(model=granite_precise)
llama_llm_creative = WatsonxLLM(model=llama_creative)
llama_llm_precise = WatsonxLLM(model=llama_precise)

# Compare responses to the same prompt
prompts = [
    "Write a short poem about artificial intelligence",
    "What are the key components of a neural network?",
    "List 5 tips for effective time management"
]

for prompt in prompts:
    print(f"\n\nPrompt: {prompt}")
    print("\nGranite Creative response (Temperature = 0.8):")
    print(granite_llm_creative.invoke(prompt))
    print("\nLlama Creative response (Temperature = 0.8):")
    print(llama_llm_creative.invoke(prompt))
    print("\nGranite Precise response (Temperature = 0.1):")
    print(granite_llm_precise.invoke(prompt))
    print("\nLlama Precise response (Temperature = 0.1):")
    print(llama_llm_precise.invoke(prompt))