from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def create_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)    


def create_summary_prompt_odata():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant that generates OData queries for SAP Business One Service Layer.

        Today's date: {today}

        Instructions:
        1. Use ONLY the entity names and fields from the provided context
        2. Match the exact entity name as shown in the context (e.g., "Orders" not "SalesOrders")
        3. Generate valid OData query using: $filter, $top, $skip, $orderby, $expand
        4. For date queries like "this month", "last week", "today", calculate from today's date above
        5. Return ONLY valid JSON, no other text

        Response format:
        {{
            "entity": "<exact entity name from context>",
            "query": {{
                "$filter": "<filter expression or null>",
                "$select": "<fields or null>",
                "$top": "<number or null>",
                "$skip": "<number or null>",
                "$orderby": "<expression or null>",
                "$expand": "<expression or null>"
            }},
            "full_url": "/<entity>?<query params>"
        }}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context:
        {context}

        Question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["context", "question", "today"],
        template=template
    )

    return prompt