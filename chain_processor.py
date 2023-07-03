from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 


class ChainProcessor:
    
    def __init__(self, temperature = 0.9, template = "", output_key = ""):
        self.llm = ChatOpenAI(temperature = temperature)
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = LLMChain(llm = self.llm, prompt = self.prompt)

    def run(self, text):
        return self.chain.run(text)


class ChainProcessorSequence:
    def __init__(self, chains = [], type = 1, verbose = True):
        print(chains)
        if type == 1:
            self.chain = SimpleSequentialChain(chains = chains, verbose = verbose)
        
        if type == 2:
            self.chain = SequentialChain(chains = chains, verbose = verbose)
    
    def run(self, text):
        return self.chain.run(text)
         
if __name__ == "__main__":
    cp1 = ChainProcessor(template = "What can be a good name for a company producing the product : {product}?", output_key = "company_name")
    #print(cp1.run("Clothes"))

    cp2 = ChainProcessor(template = "Write a description about the company in 20 words : {company_name}", output_key = "company_description")

    sc1 = ChainProcessorSequence(chains = [cp1.chain, cp2.chain], verbose = False)

    print(sc1.run("Software"))