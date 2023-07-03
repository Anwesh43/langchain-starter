from langchain.chains import LLMChain 
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 


class ChainProcessor:
    
    def __init__(self, type = 1, temperature = 0.9, template = ""):
        self.llm = ChatOpenAI(temperature = temperature)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = LLMChain(llm = self.llm, prompt = self.prompt)

    def run(self, text):
        return self.chain.run(text)


if __name__ == "__main__":
    cp1 = ChainProcessor(template = "What can be a good name for a company producing the product : {product}?")
    print(cp1.run("Clothes"))