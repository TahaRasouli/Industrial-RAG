import os
import pandas as pd
from main import *

# Load or process the collection and raw_nodes
print("Processing the file ...")
collection, raw_nodes = process_file("OPC2.xml")
print("Processing complete!!!")

df = pd.read_csv('queries_dataset.csv')

# Function to process queries for different n_results
def process_queries(df, n_results):
    sys_col = f"Sys-{n_results}"
    sim_col = f"Sim-{n_results}"
    print(f"Processing {sys_col}")
    
    for idx, q in enumerate(df["Query"]):
        results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=n_results)
        df.loc[idx, sys_col] = results[0]['rag_response']
        
        print(f"Processing {sim_col}")
        output_data = []
        if similar_nodes:
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes[:n_results], 1):
                output_data.append(
                    f"\nSimilar Node {i}: DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, "
                    f"References: {node['original_details'].get('References', 'N/A')}, "
                    f"Value: {node['original_details'].get('Value', 'N/A')}"
                )
        else:
            output_data.append("\nNo semantic similarity results available for XML nodes.")
        
        df.loc[idx, sim_col] = "\n".join(output_data)

# Process for different n_results
for n in [1, 3, 5]:
    process_queries(df, n)

# Save the DataFrame to a CSV file
df.to_csv("./evaluation.csv", index=False)
