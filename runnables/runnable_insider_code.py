import random
from abc import ABC, abstractmethod

# -------------------------------
# Base Runnable class
# -------------------------------
class runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass


# -------------------------------
# Dummy LLM
# -------------------------------
class dummyllm(runnable):
    def __init__(self):
        print("dummyllm is initialized")

    def invoke(self, prompt):
        response_list = [
            "this is first result",
            "this is second result",
            "this is the third result"
        ]
        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        return self.invoke(prompt)


# -------------------------------
# Prompt Template
# -------------------------------
class dummyprompttemplate(runnable):
    def __init__(self, prompt, input_variable):
        self.prompt = prompt
        self.input_variable = input_variable

    def invoke(self, input_dict):
        # creates final prompt string
        formatted_prompt = self.prompt.format(**input_dict)
        return formatted_prompt

    def format(self, input_dict):
        return self.prompt.format(**input_dict)


# -------------------------------
# Connector (Chain Runner)
# -------------------------------
class runnableconnector(runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        data = input_data

        for runnable_obj in self.runnable_list:
            data = runnable_obj.invoke(data)

        return data

class stroutputparser(runnable):
  def __init__(self):
    pass
    # self.input_data=input_data
  
  def invoke(self,input_data):
    return input_data['response']
# -------------------------------
# MAIN: Connecting everything
# -------------------------------

# Step 1: create prompt template
prompt = dummyprompttemplate(
    prompt="Tell me a joke about {topic}",
    input_variable=["topic"]
)

# Step 2: create LLM
llm = dummyllm()

#create output parser object
output=stroutputparser()

# Step 3: connect them in chain
chain1 = runnableconnector([prompt, llm,output])
chain2 = runnableconnector([prompt,llm])#ere we didnt include output parser in chain so it dict in output 

# Step 4: run chain
result1 = chain1.invoke({"topic": "AI"})
result2=chain2.invoke({'topic':'nikhil'})
print("\nFinal Output:")
print(result1)
print(result2)
