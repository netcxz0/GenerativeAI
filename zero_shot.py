from utils.llm_utils import llm_model

prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""
response = llm_model(prompt)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")