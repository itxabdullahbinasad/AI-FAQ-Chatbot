from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA

api_key = "AIzaSyCQZD295J1jrE3Vrzyx6llLflohKk8zEvo"

# Load the CSV file using the ISO-8859-1 encoding
llm = GooglePalm(google_api_key=api_key, temperature=0.5)

Instruct_embeddings = HuggingFaceInstructEmbeddings()

file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt", encoding="ISO-8859-1")

    # Load the data into a list
    data = loader.load()
    vector_db = FAISS.from_documents(documents=data, embedding=Instruct_embeddings)
    vector_db.save_local(file_path)


def get_qa_chain():
    vector_db = FAISS.load_local(file_path, Instruct_embeddings)
    retriever = vector_db.as_retriever(score_threshold=0.7)
    print(retriever)

    prompt_template = """Given the following context and a question, generate an answer on this content given in the document plz don't make answers on your own and if you don't see or get the answer related to the question, kindly say, I don't know
    CONTEXT: {context}
    Question: {question}"""

    Prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, input_key="query", chain_type="stuff",
                                       return_source_documents=True, chain_type_kwargs={"prompt": Prompt})
    return chain


if __name__ == "__main__":
    get_qa_chain()