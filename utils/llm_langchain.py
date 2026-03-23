import os
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

def create_llm():
    model_id = "meta-llama/llama-3-3-70b-instruct"
    project_id = "a9913486-682d-4f0c-b006-5794d01c0913"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.2,
    }
    return WatsonxLLM(
        model_id=model_id,
        url=os.environ.get("WATSONX_URL"),
        apikey=os.environ.get("WATSONX_APIKEY"),
        project_id=project_id,
        params=parameters
    )

def build_chain(template: str, llm):
    prompt = PromptTemplate.from_template(template)

    # Build chain (no need for RunnableLambda)
    return prompt | llm | StrOutputParser()


