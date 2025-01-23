import os
import pandas as pd
from main import *

print("Processing the file ...")
collection, raw_nodes = process_file("OPC2.xml")
print("Processing complete!!!")

df = pd.read_csv('queries_dataset.csv')

def process_queries(df, n_results):
    df[f"Sys-{n_results}"] = [
        {
            f'matches': [
                {
                    'rank': idx + 1,
                    'content': result['content'],
                    'node_id': result['metadata']['NodeId'] if result['metadata']['source_type'] == 'xml' else None
                }
                for idx, result in enumerate(query_documents(collection, raw_nodes, q, n_results=n_results)[0])
            ]
        }
        for q in df["Query"]
    ]
    return df

# Process for different n_results
for n in [3, 5]:
    print(f"\nProcessing queries with n_results={n}")
    df = process_queries(df, n)
    df.to_csv(f"./evaluation_{n}.csv", index=False)
