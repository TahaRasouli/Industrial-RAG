from main import query_documents, process_file
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Class to store evaluation metrics for different k values"""
    k: int
    precision: float
    recall: float
    f1_score: float
    cosine_similarity: float

def load_test_dataset(csv_path: str) -> List[Tuple[str, str]]:
    """
    Load test dataset from CSV file
   
    Args:
        csv_path: Path to the CSV file containing queries and answers
       
    Returns:
        List of (query, answer) tuples
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
       
        # Convert DataFrame to list of tuples
        test_data = list(zip(df['Query'].tolist(), df['Answer'].tolist()))
       
        # Filter out rows where the answer is "Not Found" as they won't be useful for evaluation
        test_data = [(query, answer) for query, answer in test_data if answer != "Not Found"]
       
        print(f"Loaded {len(test_data)} valid test cases from dataset")
        return test_data
       
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        return []


def calculate_similarity(response: str, true_label: str, model: SentenceTransformer) -> float:
    """
    Calculate cosine similarity between system response and true label
    """
    # Generate embeddings
    response_embedding = model.encode([response])[0]
    label_embedding = model.encode([true_label])[0]
   
    # Calculate cosine similarity
    similarity = cosine_similarity([response_embedding], [label_embedding])[0][0]
    return similarity

def calculate_metrics(relevant_count: int, retrieved_count: int, true_positives: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score
    """
    precision = true_positives / retrieved_count if retrieved_count > 0 else 0
    recall = true_positives / relevant_count if relevant_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
   
    return precision, recall, f1

def evaluate_query_results(
    query_results: List[Dict],
    true_label: str,
    k_values: List[int] = [1, 3, 5],
    similarity_threshold: float = 0.7
) -> Dict[int, EvaluationResult]:
    """
    Evaluate query results against true label for different k values
   
    Args:
        query_results: List of results from query_documents function
        true_label: The ground truth answer
        k_values: List of k values to evaluate (default: [1, 3, 5])
        similarity_threshold: Threshold for considering a match as relevant
       
    Returns:
        Dictionary mapping k values to their evaluation metrics
    """
    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
   
    evaluation_results = {}
   
    for k in k_values:
        # Get top-k results
        top_k_results = query_results[:k]
       
        # Calculate similarity scores for each result
        similarities = []
        relevant_results = 0
       
        for result in top_k_results:
            if 'rag_response' in result and result['rag_response']:
                sim_score = calculate_similarity(result['rag_response'], true_label, model)
                similarities.append(sim_score)
                if sim_score >= similarity_threshold:
                    relevant_results += 1
       
        # Calculate average similarity for top-k results
        avg_similarity = np.mean(similarities) if similarities else 0
       
        # Calculate metrics
        precision, recall, f1 = calculate_metrics(
            relevant_count=1,  # Since we have one true label
            retrieved_count=k,
            true_positives=relevant_results
        )
       
        # Store results
        evaluation_results[k] = EvaluationResult(
            k=k,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cosine_similarity=avg_similarity
        )
   
    return evaluation_results

def evaluate_test_dataset(
    test_data: List[Tuple[str, str]],
    collection,
    raw_nodes: Dict,
    k_values: List[int] = [1, 3, 5]
) -> Dict[int, List[EvaluationResult]]:
    """
    Evaluate system performance on a test dataset
   
    Args:
        test_data: List of (query, true_label) tuples
        collection: Vector database collection
        raw_nodes: Dictionary of raw node data
        k_values: List of k values to evaluate
       
    Returns:
        Dictionary mapping k values to lists of evaluation results
    """
    all_results = {k: [] for k in k_values}
   
    for query, true_label in test_data:
        # Get system results for query
        results, _ = query_documents(collection, raw_nodes, query)
       
        # Evaluate results
        evaluation = evaluate_query_results(results, true_label, k_values)
       
        # Store results
        for k in k_values:
            all_results[k].append(evaluation[k])
   
    return all_results

def print_evaluation_summary(evaluation_results: Dict[int, List[EvaluationResult]]):
    """
    Print summary of evaluation results
    """
    print("\nEvaluation Summary:")
    print("=" * 50)
   
    for k, results in evaluation_results.items():
        avg_precision = np.mean([r.precision for r in results])
        avg_recall = np.mean([r.recall for r in results])
        avg_f1 = np.mean([r.f1_score for r in results])
        avg_similarity = np.mean([r.cosine_similarity for r in results])
       
        print(f"\nTop-{k} Results:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Cosine Similarity: {avg_similarity:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    # Load the document to be queried
    file_path = input("Enter the path to your PDF or XML file: ")
    print("\nProcessing document...")
   
    # Process the file and get raw nodes
    collection, raw_nodes = process_file(file_path)
   
    # Load test dataset
    dataset_path = input("Enter the path to your queries_dataset.csv file: ")
    test_dataset = load_test_dataset(dataset_path)
   
    if test_dataset:
        # Run evaluation
        print("\nRunning evaluation...")
        evaluation_results = evaluate_test_dataset(test_dataset, collection, raw_nodes)
       
        # Print results
        print_evaluation_summary(evaluation_results)
    else:
        print("No valid test cases found in the dataset.")
