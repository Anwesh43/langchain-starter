from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory

class WithMemoryChatProcessor: 
    
    def __init__(self, verbose=False, temperature=0.4):
        self.llm = ChatOpenAI(temperature=temperature)
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm = self.llm, memory = self.memory, verbose = verbose)
    
    def chat(self, sentence):
        return self.chain.predict(input = sentence)

    def get_memory_variables(self):
        return self.memory.load_memory_variables({})
    
    def get_buffer(self):
        return self.memory.buffer 

    def reinitialize_memory(self, inputSentence, outputSentence):
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm = self.llm, memory = self.memory, verbose = False)
        self.memory.save_context({"input":inputSentence}, {"output": outputSentence})
    


if __name__ == "__main__":
    wmcp = WithMemoryChatProcessor()
    print(wmcp.chat("Hi I am Anwesh"))
    print(wmcp.chat("My profession is a software developer"))
    print(wmcp.chat("What is my name?"))
    print(wmcp.chat("What is my profession?"))
    print(wmcp.get_memory_variables())
    print("Re initializing memory")
    print(wmcp.reinitialize_memory("My name is Shaktiman, please initialize me an hero_id", "Hello Shaktimaan! your hero_id is 1239812"))
    print(wmcp.get_buffer())
    print(wmcp.chat("What is my hero_id?"))
    print(wmcp.chat("What is my name?"))