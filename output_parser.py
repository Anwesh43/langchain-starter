from chat_prompt_template import TemplateProcessor
from langchain.output_parsers import ResponseSchema, StructuredOutputParser 



def get_response_schema(name, description):
    return ResponseSchema(name=name, description = description)

gift_schema = get_response_schema("gift",
                             "Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = get_response_schema("delivery_days",
                                      "How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = get_response_schema("price_value",
                                    "Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

schemas = [gift_schema, delivery_days_schema, price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(schemas)

format_instruction = output_parser.get_format_instructions()

print(format_instruction)

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}

{format_instructions}
"""



tp = TemplateProcessor(review_template)
print(tp.get_variables())

response = tp.format_and_chat(text = customer_review, format_instructions = format_instruction)
result_dict = output_parser.parse(response.content)
print(result_dict)
print(result_dict.get('delivery_days'))