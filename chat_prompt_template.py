from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# def templatePrompt(template):
#     prompt_template = ChatPromptTemplate.from_template(template)
#     print(prompt_template.messages[0].prompt)
#     print(prompt_template.messages[0].prompt.input_variables)
#     return prompt_template 

class TemplateProcessor:
    
    def __init__(self, template):
        self.prompt_template = ChatPromptTemplate.from_template(template)
        self.chat = ChatOpenAI()
    
    def get_variables(self):
        print(self.prompt_template.messages[0].prompt.input_variables)
    
    def format_and_chat(self, **kwargs):
        print(kwargs)
        messages = self.prompt_template.format_messages(**kwargs)
        response = self.chat(messages)
        return response 

        
if __name__ == "__main__":
    template_string = "Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"
    # chat = ChatOpenAI()
    # print(chat)
    pt = TemplateProcessor(template_string)
    print(pt.get_variables())

    customer_style = "American English in a calm and respectful tone"
    customer_email = "Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!"


    print("Getting customer response")

    response = pt.format_and_chat(style = customer_style, text = customer_email)

    print(response.content)

    service_reply = """Hey there customer, \
    the warranty does not cover \
    cleaning expenses for your kitchen \
    because it's your fault that \
    you misused your blender \
    by forgetting to put the lid on before \
    starting the blender. \
    Tough luck! See ya!
    """

    service_style_pirate = """\
    a polite tone \
    that speaks in English Pirate\
    """

    print("Getting Pirate response")
    service_response = pt.format_and_chat(style = service_style_pirate, text = service_reply)
    print(service_response.content)
