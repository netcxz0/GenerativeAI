import warnings
warnings.filterwarnings('ignore')

from langchain_ibm import WatsonxLLM

granite_llm = WatsonxLLM(    model_id=”ibm/granite-3-2-8b-instruct”,    url=”https://us-south.ml.cloud.ibm.com",    project_id=”skills-network”,    params={        “max_new_tokens”: 256,        “temperature”: 0.5,        “top_p”: 0.2    })

def llm_model(prompt_txt, params=None):
    model_id = "ibm/granite-3-2-8b-instruct"
    default_params = {
        "max_new_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.2
    }
    if params:
        default_params.update(params)

granite_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=default_params
)

response = granite_llm.invoke(prompt_txt)
return response</code></pre>
