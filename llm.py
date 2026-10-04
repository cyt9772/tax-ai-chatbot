from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langchain_classic.chains import RetrievalQA



def get_ai_message(user_message):
    embedding=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)
    index_name='tax-index'
    database=PineconeVectorStore.from_existing_index(embedding=embedding,index_name=index_name)

    llm=ChatOpenAI(model='gpt-4o')
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
    retriever=database.as_retriever(search_kwargs={'k':4})

    qa_chain=RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={'prompt':prompt}
    )

    dictionary=["사람을 나타내는 표현 -> 거주자"]
    prompt=ChatPromptTemplate.from_template(f"""
                사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
                만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
                그런 경우에는 질문만 return해 주세요
                사전: {dictionary}
                
                질문: {{question}}""")
    dictionary_chain= prompt|llm|StrOutputParser()
    tax_chain={"query":dictionary_chain}|qa_chain
    ai_message=tax_chain.invoke({"question":user_message})
    return ai_message['result']
