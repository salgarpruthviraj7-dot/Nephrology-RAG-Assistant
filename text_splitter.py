from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    pass

