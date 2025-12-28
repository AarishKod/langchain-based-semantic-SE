from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
import os

# loading environment variables
load_dotenv()

# api keys loaded via os.getenv()
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING: str | None = os.getenv("LANGSMITH_TRACING")

# path of document being analyzed (Nike form 10-k)
FILE_PATH = "./data/nke-10k-2023.pdf"

# defining loader
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()

# creating text splitter with overlap 200 and chunk size 1000 (chars)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# splitting first 3 pages of Nike form 10-k
all_splits = text_splitter.split_documents(docs)

"""
embeddings are numerical representations of text that capture meaning. It's a conversion of words, sentences,
or sometimes documents into a list of vectors that computers can use to mathematically compare with one another.
3 dimensional example: vectors for "car" may be [0.8, 0.3, 0.1]. Vectors for "automobile" then could be [0.81, 0.29, 0.09]. 
This tells the computer that "car" and "automobile" must be similar. In the case of semantic search, we can find documents
not just by matching keywords but also by meaning. If we were to search up "How to fix a flat tire" it would match with 
"replacing a punctured wheel" because the vectors would be similar. This makes semantic search much more powerful and
useful than keyword search. 
"""

# creating embeddings. Using HuggingFace because embeddings will be calculated locally on cpu.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# creating a langchain vector store object and initializing it with the HuggingFace embedding store
# this tells the in memory vector store what embeddings model to use
vector_store = InMemoryVectorStore(embedding=embeddings)

# we then give the vector store documents to embed. In this case, that's our 1000 char long 
# 200 char overlap chunks
ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search_with_score(
    "2023 net profit"
)

for result in results:
    print(result)
    print("______________________________________________________")





