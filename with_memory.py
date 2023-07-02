from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory

class WithMemoryChatProcessor: 
    
    def __init__(self, verbose=False, temperature=0.4, type = 1, k = 1, max_token=50):
        self.llm = ChatOpenAI(temperature=temperature)
        if type == 1:
            self.memory = ConversationBufferMemory()
        if type == 2:
            self.memory = ConversationBufferWindowMemory(k = k)
        if type == 3:
            self.memory = ConversationTokenBufferMemory(llm = self.llm, max_token_limit = max_token)
        if type == 4:
            self.memory = ConversationSummaryBufferMemory(llm = self.llm, max_token_limit = max_token)

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
    
    def feed(self, inputSentence, outputSentence):
        self.memory.save_context({"input": inputSentence}, {"output": outputSentence})


if __name__ == "__main__":
    wmcp = WithMemoryChatProcessor(type=2)
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

    wmcp2 = WithMemoryChatProcessor(type = 3)
    print(wmcp2.chat("AI is What?"))
    print(wmcp2.chat("Backpropagation is what?"))
    print(wmcp2.chat("What is Deep Learning?"))
    print(wmcp2.chat("What is Neural Networks?"))
    # print(wmcp2.get_buffer())

    wmcp3 = WithMemoryChatProcessor(type=4, max_token = 100)
    wmcp3.feed("Hello", "What's up")
    wmcp3.feed("Not much, just hanging", "Cool")
    wmcp3.feed("What is on the schedule today?", """"There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo.""")
    print(wmcp3.get_memory_variables())
    print(wmcp3.chat("What would be a good demo to show?"))
    print(wmcp3.get_memory_variables())