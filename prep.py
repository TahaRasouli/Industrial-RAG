import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import re
import numpy as np
import pandas as pd
import csv
import time
from datetime import datetime

# Original XML processing functions remain unchanged
def extract_node_details(element):
    """
    Extracts details like description, value, NodeId, DisplayName, and references from an XML element.
    """
    details = {
        "NodeId": element.attrib.get("NodeId", "N/A"),
        "Description": None,
        "DisplayName": None,
        "Value": None,
        "References": []
    }
    for child in element:
        tag = child.tag.split('}')[-1]
        if tag == "Description":
            details["Description"] = child.text
        elif tag == "DisplayName":
            details["DisplayName"] = child.text
        elif tag == "Value":
            details["Value"] = extract_value_content(child)
        elif tag == "References":
            for reference in child:
                if reference.tag.split('}')[-1] == "Reference":
                    # Append both attributes and the text value of the reference
                    reference_details = reference.attrib.copy()
                    reference_details["Value"] = reference.text.strip() if reference.text else "N/A"
                    details["References"].append(reference_details)
    return details


def extract_value_content(value_element):
    """
    Recursively extracts the content of a <Value> element, handling any embedded child elements.
    """
    if not list(value_element):  # No child elements, return text directly
        return value_element.text or "No value provided."
    # Process child elements
    content = []
    for child in value_element:
        tag = child.tag.split('}')[-1]
        child_text = child.text.strip() if child.text else ""
        content.append(f"<{tag}>{child_text}</{tag}>")
    return "".join(content)

def parse_nodes_to_dict(filename):
    """
    Parses the XML file and saves node details into a dictionary.
    Each node's NodeId serves as the key, and the value is a dictionary of the node's details.
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    # Retrieve namespace from the root
    namespace = root.tag.split('}')[0].strip('{')
    # Node types to extract
    node_types = ["UAObject", "UAVariable", "UAObjectType"]
    nodes_dict = {}
    for node_type in node_types:
        for element in root.findall(f".//{{{namespace}}}{node_type}"):
            details = extract_node_details(element)
            node_id = details["NodeId"]
            if node_id != "N/A":
                nodes_dict[node_id] = details
    return nodes_dict


def format_node_content(details):
    """
    Formats raw node details into a single string for semantic comparison.
    """
    content_parts = []
    
    if details["Description"]:
        content_parts.append(f"Description: {details['Description']}")
    if details["DisplayName"]:
        content_parts.append(f"DisplayName: {details['DisplayName']}")
    if details["Value"]:
        content_parts.append(f"Value: {details['Value']}")
    
    return " | ".join(content_parts)
    

def convert_to_natural_language(details):
    """
    Converts node details to natural language using Groq LLM.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [
        {
            "role": "user",
            "content": f"Convert the following node details to natural language: {details}"
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# New file type detection and processing functions without magic library
def detect_file_type(file_path):
    """
    Detects if the input file is PDF or XML using file extension and content analysis.
    """
    try:
        # Check file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Read the first few bytes of the file to check its content
        with open(file_path, 'rb') as f:
            header = f.read(8)  # Read first 8 bytes
            
        # Check for PDF signature
        if file_extension == '.pdf' or header.startswith(b'%PDF'):
            # Verify it's actually a PDF by trying to open it
            try:
                with open(file_path, 'rb') as f:
                    PyPDF2.PdfReader(f)
                return 'pdf'
            except:
                return 'unknown'
        
        # Check for XML
        elif file_extension == '.xml':
            # Try to parse as XML
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_start = f.read(1024)  # Read first 1KB
                    # Check for XML declaration or root element
                    if content_start.strip().startswith(('<?xml', '<')):
                        ET.parse(file_path)  # Verify it's valid XML
                        return 'xml'
            except:
                return 'unknown'
        
        return 'unknown'
        
    except Exception as e:
        print(f"Error detecting file type: {str(e)}")
        return 'unknown'

def process_pdf(file_path):
    """
    Extracts text content from PDF and splits it into meaningful chunks.
    """
    try:
        chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Split text into paragraphs
                paragraphs = text.split('\n\n')
                
                # Process each paragraph
                for para_num, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) > 0:  # Skip empty paragraphs
                        chunk = {
                            'content': paragraph.strip(),
                            'metadata': {
                                'page_number': page_num + 1,
                                'paragraph_number': para_num + 1,
                                'source_type': 'pdf',
                                'file_name': os.path.basename(file_path)
                            }
                        }
                        chunks.append(chunk)
                        
        return chunks
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []

def add_to_vector_db(collection, chunks, embedder):
    """
    Adds processed chunks to the vector database with proper metadata.
    """
    try:
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{chunk['metadata']['file_name']}_{chunk['metadata']['page_number']}_{chunk['metadata']['paragraph_number']}"
            
            collection.add(
                documents=[chunk['content']],
                metadatas=[chunk['metadata']],
                ids=[chunk_id]
            )
            
    except Exception as e:
        print(f"Error adding to vector database: {str(e)}")


def process_file(file_path):
    """
    Main function to process either PDF or XML file and add to vector database.
    Also returns the raw node details for XML files.
    """
    try:
        # Initialize ChromaDB and embedding function
        client = chromadb.Client()
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        collection = client.create_collection(
            name="document_embeddings",
            get_or_create=True
        )
        
        # Store for raw node details
        raw_nodes = {}
        
        # Detect file type
        file_type = detect_file_type(file_path)
        
        if file_type == 'pdf':
            # Process PDF
            chunks = process_pdf(file_path)
            add_to_vector_db(collection, chunks, embedder)
            
        elif file_type == 'xml':
            # Parse XML and store raw nodes
            raw_nodes = parse_nodes_to_dict(file_path)
            
            # Convert to natural language for RAG
            for node_id, details in raw_nodes.items():
                nl_description = convert_to_natural_language(details)
                
                # Add to vector DB
                collection.add(
                    documents=[nl_description],
                    metadatas=[{"NodeId": node_id, "source_type": "xml"}],
                    ids=[node_id]
                )
                
        else:
            raise ValueError("Unsupported file type")
            
        return collection, raw_nodes
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, {}


def generate_rag_response(query_text, context):
    """
    Generates a RAG response using the Groq LLM based on the query and retrieved context.
    
    Args:
        query_text (str): The user's query
        context (str): The retrieved context from the vector database
        
    Returns:
        str: The generated response from the LLM
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. "
                          "If the context doesn't contain relevant information, acknowledge that."
            },
            {
                "role": "user",
                "content": f"Answer the following query based on the provided context:\n\n"
                          f"Query: {query_text}\n\n"
                          f"Context: {context}"
            }
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating RAG response: {str(e)}")
        return "Error generating response"


def find_similar_nodes(query_text, raw_nodes, top_k=5):
    """
    Finds the most semantically similar nodes to the query using raw node content.
    
    Args:
        query_text (str): The user's query
        raw_nodes (dict): Dictionary of node_id: node_details pairs
        top_k (int): Number of top results to return
    """
    try:
        # Initialize the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Format node contents and create mapping
        node_contents = {}
        for node_id, details in raw_nodes.items():
            formatted_content = format_node_content(details)
            if formatted_content:  # Only include nodes with content
                node_contents[node_id] = formatted_content
        
        # Generate embeddings for the query
        query_embedding = model.encode([query_text])[0]
        
        # Create a list of (node_id, content) tuples
        nodes = list(node_contents.items())
        contents = [content for _, content in nodes]
        
        # Generate embeddings for all node contents
        content_embeddings = model.encode(contents)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], content_embeddings)[0]
        
        # Get indices of top-k similar nodes
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            node_id, content = nodes[idx]
            similarity_score = similarities[idx]
            results.append({
                'node_id': node_id,
                'raw_content': content,
                'original_details': raw_nodes[node_id],
                'similarity_score': similarity_score
            })
            
        return results
        
    except Exception as e:
        print(f"Error finding similar nodes: {str(e)}")
        return []


def query_documents(collection, raw_nodes, query_text, n_results=5):
    """
    Query the vector database and perform semantic similarity search on raw nodes.
    """
    try:
        # Get results from vector database
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Combine the retrieved results into context for RAG
        retrieved_context = "\n".join(results["documents"][0])
        
        # Generate RAG response
        rag_response = generate_rag_response(query_text, retrieved_context)
        
        # Find semantically similar nodes using raw node content
        similar_nodes = find_similar_nodes(query_text, raw_nodes) if raw_nodes else []
        
        # Format vector DB results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i] if "distances" in results else None,
                "rag_response": rag_response if i == 0 else None
            }
            formatted_results.append(result)
            
        return formatted_results, similar_nodes
        
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        return [], []

def evaluate_queries(queries_df, collection, raw_nodes):
    results = []
    call_count = 0
    
    for idx, row in queries_df.iterrows():
        query = row['Query']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Processing query {idx + 1}/{len(queries_df)}: {query}")
        
        # Rate limiting
        if call_count >= 1000:
            print(f"[{current_time}] Reached 1000 calls. Resting for 1 minute...")
            time.sleep(60)
            call_count = 0
        
        vector_results, similar_nodes = query_documents(collection, raw_nodes, query)
        call_count += 1  # Increment for RAG call
        
        sys_1 = vector_results[0]['rag_response'] if vector_results and vector_results[0]['rag_response'] else ''
        
        sys_3 = []
        sys_5 = []
        for i, result in enumerate(vector_results[:5]):
            result_str = f"Content: {result['content']}, NodeId: {result['metadata'].get('NodeId', 'N/A')}"
            if i < 3:
                sys_3.append(result_str)
            sys_5.append(result_str)
        
        sim_1 = ''
        sim_3 = []
        sim_5 = []
        
        for i, node in enumerate(similar_nodes[:5]):
            node_str = (f"NodeId: {node['node_id']}, Description: {node['original_details'].get('Description', 'N/A')}, "
                       f"DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, "
                       f"Value: {node['original_details'].get('Value', 'N/A')}, "
                       f"Score: {node['similarity_score']:.4f}")
            
            if i == 0:
                sim_1 = node_str
            if i < 3:
                sim_3.append(node_str)
            sim_5.append(node_str)
        
        results.append({
            'Query': query,
            'Answer': row['Answer'],
            'Sys-1': sys_1,
            'Sys-3': ' | '.join(sys_3),
            'Sys-5': ' | '.join(sys_5),
            'Sim-1': sim_1,
            'Sim-3': ' | '.join(sim_3),
            'Sim-5': ' | '.join(sim_5)
        })
    
    return pd.DataFrame(results)

def run_evaluation(queries_file, output_file):
    print(f"Loading queries from {queries_file}")
    queries_df = pd.read_csv(queries_file)
    
    print("Processing input file...")
    collection, raw_nodes = process_file('OPC2.xml')
    
    if collection:
        print("Starting evaluation...")
        results_df = evaluate_queries(queries_df, collection, raw_nodes)
        results_df.to_csv(output_file, index=False)
        print(f"Evaluation complete. Results saved to {output_file}")
    else:
        print("Error processing input file")

run_evaluation('queries_dataset.csv', 'evaluation.csv')
