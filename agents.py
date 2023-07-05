
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

class AgentRunner:
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature = 0)
    
    def create_agent(self, tools):
        self.tools = load_tools(tools, llm = self.llm)
        self.agent = initialize_agent(
            self.tools, 
            self.llm,  
            agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_erros = True, 
            verbose = True 
        )
        print(self.agent) 

    def run(self, text):
        self.agent(text)


if __name__ == "__main__":
    agentRunner = AgentRunner()
    agentRunner.create_agent(["llm-math", "wikipedia"])
    agentRunner.run("Tom M. Mitchell is an American computer scientist \
    and the Founders University Professor at Carnegie Mellon University (CMU)\
    what book did he write?")