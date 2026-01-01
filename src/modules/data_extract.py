from langchain_community.document_loaders import TextLoader

def text_to_vector(filepath):
    file_loader = TextLoader(filepath)
    documents = file_loader.load()
    # Combine all document contents into a single string
    return "\n".join(doc.page_content for doc in documents)
