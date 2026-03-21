# IBM WatsonX imports
# from ibm_watsonx_ai.foundation_models import Model
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
import os
from langchain_ibm import WatsonxLLM
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableSequence
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.chains import LLMChain  # Still using this for backward compatibility

def llm_model(prompt_txt, params=None):
    model_id = "ibm/granite-3-3-8b-instruct"
    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    project_id = "a9913486-682d-4f0c-b006-5794d01c0913"

    # Create LLM directly
    granite_llm = WatsonxLLM(
        model_id=model_id,
        url=os.environ.get("WATSONX_URL"),
        apikey=os.environ.get("WATSONX+APIKEY"),
        project_id=project_id,
        params=default_params
    )

    response = granite_llm.invoke(prompt_txt)
    return response
