# from celery import shared_task
# from .models import Document
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import TokenTextSplitter
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_community.vectorstores import Neo4jVector
#
# @shared_task
# def process_document(document_id):
#     document = Document.objects.get(id=document_id)
#
#     # Load and process the PDF
#     loader = PyPDFLoader(document.file.path)
#     pages = loader.load()
#
#     text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
#     splits = text_splitter.split_documents(pages)
#
#     # Create graph documents
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
#     llm_transformer = LLMGraphTransformer(llm=llm)
#     graph_documents = llm_transformer.convert_to_graph_documents(splits)
#
#     # Add to Neo4j graph
#     graph = Neo4jGraph()
#     graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
#
#     # Create vector index
#     vector_index = Neo4jVector.from_existing_graph(
#         OpenAIEmbeddings(),
#         search_type="hybrid",
#         node_label="Document",
#         text_node_properties=["text"],
#         embedding_node_property="embedding"
#     )
#
#     document.processed = True
#     document.save()