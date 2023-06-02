from langchain.llms import OpenAI
import sys 
llm = OpenAI(temperature=0.9)
print(llm("".join(sys.argv[1:])))