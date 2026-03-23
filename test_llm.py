def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from utils.inference_llm import inference_llm
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = inference_llm()
# msg = model.generate("In today's sales meeting, we ")
# print(msg['results'][0]['generated_text'])

llama_llm = WatsonxLLM(model = model)

msg = llama_llm.invoke(
     [
        SystemMessage(content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence"),
        HumanMessage(content="I enjoy mystery novels, what should I read?")
    ]
)

print(msg)