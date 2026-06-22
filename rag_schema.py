class RAGPipeline:
    LESSON_BASE_URL = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/"
    INSTRUCTIONS = '''
        Your task is to answer questions from the course participants
        based on the provided context.
        
        Use the context to find relevant information and provide accurate
        answers. If the answer is not found in the context,
        respond with "I don't know.'''.strip()

    # The __init__ runs at instatiation of a new RAGPipeline object
    def __init__(
        self,
        index_base,
        llm_client
    ):
        # Store parameters on the instance so other methods can access them
        self.index_base = index_base
        # The llm_client is an instance of the OpenAI class
        self.llm_client = llm_client
    # End of __init__
    
    # I define the Class' methods
    def search(self, user_query, num_results=5) -> list:
        """
        This method receive a question from the user and perform a search in the knowledge base.
        By default it returns the 5 most relevant results.
        """
        search_results = self.index_base.search(user_query, num_results=num_results)
        return search_results
    
    def build_context(self, search_results: list) -> str:
        """
        This method is a helper to the build_prompt method.
        It receice a list of search results and it extracts the content to create a context to include in the prompt.
        It returns a string of 
        """
        prompt_context = ""

        # The enumerate function returns a tuple containing the index and the value of the element in any iterable.
        for doc_num, lesson_doc in enumerate(search_results, start=1):
            prompt_context += (
                f"Document #{doc_num}\n"
                f"Filename: {lesson_doc['filename']}\n"
                f"Lesson URL: {self.LESSON_BASE_URL}{lesson_doc['filename']}\n"
                f"Content: {lesson_doc['content']}\n\n"
            )
        return prompt_context

    def build_prompt(self, prompt_context: str, user_query: str) -> str:
        """
        This method creates a prompt to send to the LLM to elaborate an answer to the user's question.
        Give the stateless nature of LLM, we provide three important elements:
        1. Instruction to the LLM to behave as we like it. 
        2. A context it needs to be aware of, which is a list of lessons related to the user's questions.
        3. The actual question asked by the user that needs to get answered. 
        """
        prompt = (
            f"CONTEXT: {prompt_context}\n\n"
            f"QUESTION: {user_query}\n\n"
        ).strip()
        
        return prompt
    
    def answer(self, prompt: str) -> tuple[str, dict]:
        """
        This method sends the prompt to the LLM and returns the answer.
        Also it return the token usage to generate the answer.
        The output_text is the answer from the LLM.
        The usage is a dictionary containing the token usage for the prompt and the answer.
        """
        input_messages = [
            {"role": "developer", "content": self.INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model="gpt-5.4-mini",
            input=input_messages
        )
        return response.output_text, response.usage
    
    def execute_rag(self, user_query: str) -> tuple[str, dict]:
        """
        This method executes the RAG pipeline for a given user query.
        It first performs a search in the knowledge base, then it builds a context and a prompt.
        Finally it sends the prompt to the LLM to generate an answer.
        It returns the answer and the token usage.
        """
        search_results = self.search(user_query)
        prompt_context = self.build_context(search_results)
        prompt = self.build_prompt(prompt_context, user_query)
        answer, usage = self.answer(prompt) # this line unpacks the tuple returned by the answer method into two variables: answer and usage
        return answer, usage