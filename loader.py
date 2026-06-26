from langchain_community.document_loaders import PyPDFLoader

def document_loader(path):
    try:
        loader =PyPDFLoader(path)
        return loader.load()
    except:
        return "File not Found in the directory"
if __name__ == "__main__":
    documents = document_loader("comprehensive-clinical-nephrology.pdf")