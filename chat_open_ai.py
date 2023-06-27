from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0.0)

print(chat)
response = chat.predict("What is 1 + 1?")

print(response)