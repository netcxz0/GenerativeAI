from utils.llm_langchain import build_chain
from utils.llm_langchain import create_llm

template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

review = "I love this smartphone! The camera is amazing but it heats up."

llm = create_llm()
chain = build_chain(template, llm)
result = chain.invoke({"review": review})

print(result)