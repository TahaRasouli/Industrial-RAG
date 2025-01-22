import os
import pandas as pd
from main import *

# Load or process the collection and raw_nodes
print("Processing the file ...")
collection, raw_nodes = process_file("OPC2.xml")
print("Processing complete!!!")

# Now you can proceed with the rest of the script as before
df = pd.read_csv('queries_dataset.csv')

print("Processing Sys-1")
for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=1)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-1"] = results[0]['rag_response']
    print("Processing Sim-1")
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append(f"\nSimilar Node {i}: DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, "
                                   f"References: {node['original_details'].get('References', 'N/A')}, "
                                   f"Value: {node['original_details'].get('Value', 'N/A')}")
            df.loc[idx, "Sim-1"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-1"] = "\n".join(output_data)

print("Processing Sys-3")
for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=3)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-3"] = results[0]['rag_response']
    print("Processing Sim-3")
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append(f"\nSimilar Node {i}: DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, "
                                   f"References: {node['original_details'].get('References', 'N/A')}, "
                                   f"Value: {node['original_details'].get('Value', 'N/A')}")
            df.loc[idx, "Sim-3"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-3"] = "\n".join(output_data)


print("Processing Sys-5")
for idx, q in enumerate(df["Query"]):
    results, similar_nodes = query_documents(collection, raw_nodes, q, n_results=5)
    
    # Add the result to the corresponding row in a new column
    df.loc[idx, "Sys-5"] = results[0]['rag_response']
    print("Processing Sim-5")
    if similar_nodes:
        for node in similar_nodes:
            output_data = []
            output_data.append("\nTop Semantically Similar Nodes (Raw Content):")
            for i, node in enumerate(similar_nodes, 1):
                output_data.append(f"\nSimilar Node {i}: DisplayName: {node['original_details'].get('DisplayName', 'N/A')}, "
                                   f"References: {node['original_details'].get('References', 'N/A')}, "
                                   f"Value: {node['original_details'].get('Value', 'N/A')}")
            df.loc[idx, "Sim-5"] = "\n".join(output_data)
    else:
        output_data.append("\nNo semantic similarity results available for XML nodes.")
        df.loc[idx, "Sim-5"] = "\n".join(output_data)
        
# Save the DataFrame to a CSV file
df.to_csv("./evaluation.csv", index=False)
