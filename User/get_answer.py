from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def get_response1(llm, vectorstore, question):
    # Step 1: Enhance the question for clarity
    question_enhanced_prompt = """
    Human: Please enhance the following question to make it
    clearer and more specific if needed.
    Question: {question}
    Enhanced Question:
    """
    enhanced_question_prompt = PromptTemplate(
        template=question_enhanced_prompt, input_variables=["question"]
    )

    # Send the prompt to the LLM and get the enhanced version of the question
    enhanced_question = llm(
        enhanced_question_prompt.format(question=question)).strip()

    # Step 2: Define a prompt template for retrieval-based QA
    prompt_template = """
    Human: Please use the given context to provide a concise answer
    to the question. If you don't know the answer, just say that
    you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Step 3: Set up a RetrievalQA chain to handle retrieval process
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple retrieval-based chain type
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        ),
        # Include retrieved documents for validation
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    # Get the answer from the LLM using the enhanced question
    answer = qa({"query": enhanced_question})

    return answer["result"]


# def get_response2(llm, vectorstore, question):
#     # Step 1: Generate 5 distinct answers without vector search
#     prompt_template = """
#     Human: Please provide 5 distinct answers to the following question.
#     Each answer should be concise and clear. If you don't know the answer
#     to a question, just say that you don't know and don't try
#     to make up an answer.
#     Question: {question}
#     Assistant:
#     1.
#     2.
#     3.
#     4.
#     5.
#     """
#     prompt = PromptTemplate(
#         template=prompt_template, input_variables=["question"])

#     response = llm(prompt.format(question=question))
#     answers = response.split("\n")
#     answers = [answer.strip() for answer in answers if answer.strip()]

#     # Ensure there are exactly 5 answers
#     while len(answers) < 5:
#         answers.append("I don't know the answer.")

#     # Step 2: Use the answers to perform a vector search
#     context_for_search = "\n".join(answers)

#     retriever = vectorstore.as_retriever(
#         search_type="similarity", search_kwargs={"k": 5}
#     )
#     docs = retriever.get_relevant_documents(context_for_search)

#     # Display the retrieved documents
#     st.write(docs)

#     # Step 3: Refine the answers using retrieved context
#     context = "\n".join([doc.page_content for doc in docs])

#     refine_prompt = """
#     Human: Here are 5 answers provided by the model.
#     Please review them in light of the following context from
#     relevant documents. If any of the answers seem incorrect,
#     adjust them based on the context provided below.
#     If an answer is already correct, leave it as is.

#     Context:
#     {context}

#     Answers:
#     1. {answer1}
#     2. {answer2}
#     3. {answer3}
#     4. {answer4}
#     5. {answer5}

#     Adjusted Answers:
#     """
#     refined_prompt = PromptTemplate(
#         template=refine_prompt,
#         input_variables=[
#             "context",
#             "answer1",
#             "answer2",
#             "answer3",
#             "answer4",
#             "answer5",
#         ],
#     )

#     refined_response = llm(
#         refined_prompt.format(
#             context=context,
#             answer1=answers[0],
#             answer2=answers[1],
#             answer3=answers[2],
#             answer4=answers[3],
#             answer5=answers[4],
#         )
#     )

#     refined_answers = refined_response.split("\n")
#     refined_answers = [
#       answer.strip() for answer in refined_answers if answer.strip()]

#     return refined_answers[:5]
