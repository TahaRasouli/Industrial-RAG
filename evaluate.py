import os
import pandas as pd
from main import *

# Load or process the collection and raw_nodes
print("Processing the file ...")
collection, raw_nodes = process_file("OPC2.xml")
print("Processing complete!!!")

# Now you can proceed with the rest of the script as before
df = pd.read_csv('queries_dataset.csv')

for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=1)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-1"] = results[0]['rag_response']
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append((f"\nSimilar Node {i}:"))
                output_data.append((f"DisplayName: {node['original_details'].get('DisplayName', 'N/A')}"))
                output_data.append((f"References: {node['original_details'].get('References', 'N/A')}"))
                output_data.append((f"Similarity Score: {node['similarity_score']:.4f}"))
                df.loc[idx, "Sim-1"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-1"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)


for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=3)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-3"] = results[0]['rag_response']
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append((f"\nSimilar Node {i}:"))
                output_data.append((f"DisplayName: {node['original_details'].get('DisplayName', 'N/A')}"))
                output_data.append((f"References: {node['original_details'].get('References', 'N/A')}"))
                output_data.append((f"Value: {node['original_details'].get('Value', 'N/A')}"))
                df.loc[idx, "Sim-3"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-3"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)


for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=5)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-5"] = results[0]['rag_response']
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append((f"\nSimilar Node {i}:"))
                output_data.append((f"DisplayName: {node['original_details'].get('DisplayName', 'N/A')}"))
                output_data.append((f"References: {node['original_details'].get('References', 'N/A')}"))
                output_data.append((f"Value: {node['original_details'].get('Value', 'N/A')}"))
                df.loc[idx, "Sim-5"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-5"] = df.loc[idx, "Sim-1"] = "\n".join(output_data)
        
# Save the DataFrame to a CSV file
df.to_csv("./evaluation.csv", index=False)
