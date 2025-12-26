from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./data/nke-10k-2023.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

# print(f"{docs[0].page_content}\n")
# print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs[0:3])

for split in all_splits:
    print(f"{split}\n")
    print("_______________________________________________________")