import os
import pandas as pd
from main import *

print("Processing the file ...")
collection, raw_nodes = process_file("OPC2.xml")
print("Processing complete!!!")

df = pd.read_csv('queries_dataset.csv')

def process_queries(df, n_results):
    sys_col = f"Sys-{n_results}"
    sim_col = f"Sim-{n_results}"
    
    for idx, q in enumerate(df["Query"]):
        results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=n_results)
        
        # Process RAG response
        if results and results[0]['rag_response']:
            df.loc[idx, sys_col] = results[0]['rag_response']
        
        # Process similar nodes
        output_data = []
        if similar_nodes and len(similar_nodes) > 0:
            output_data.append("Top Semantically Similar Nodes (Raw Content):")
            # Only take the first n_results nodes
            for i, node in enumerate(similar_nodes[:n_results], 1):
                node_info = (
                    f"\nSimilar Node {i}: NodeId: {node['node_id']}, 
                    DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, 
                    References: {node['original_details'].get('References', 'N/A')}, 
                    Value: {node['original_details'].get('Value', 'N/A')}"
                )
                output_data.append(node_info)
        else:
            output_data.append("No semantic similarity results available for XML nodes.")
        
        df.loc[idx, sim_col] = "\n".join(output_data)
        print(f"Processed query {idx+1} for {sys_col} and {sim_col}")

# Process for different n_results
for n in [1, 3, 5]:
    print(f"\nProcessing queries with n_results={n}")
    process_queries(df, n)

df.to_csv("./evaluation.csv", index=False)
