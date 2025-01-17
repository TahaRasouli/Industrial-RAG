from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from collections import defaultdict
from main import *
import pickle
import hashlib
from pathlib import Path



# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

@dataclass
class DetailedMetrics:
    """Comprehensive evaluation metrics for a single query result"""
    semantic_similarity: float
    bm25_score: float
    structural_similarity: Optional[float]  # Only for XML
    relevance_score: float
    response_quality: float
    combined_score: float

@dataclass
class EvaluationResult:
    """Extended evaluation results for different k values"""
    k: int
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    mrr: float
    detailed_metrics: List[DetailedMetrics]
    
class EnhancedEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 scoring"""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        embedding1 = self.model.encode([text1])[0]
        embedding2 = self.model.encode([text2])[0]
        return float(cosine_similarity([embedding1], [embedding2])[0][0])
        
    def calculate_bm25_score(self, query: str, document: str) -> float:
        """Calculate BM25 relevance score"""
        tokenized_query = self.preprocess_text(query)
        tokenized_doc = self.preprocess_text(document)
        
        # Create BM25 object with single document
        bm25 = BM25Okapi([tokenized_doc])
        return float(bm25.get_scores(tokenized_query)[0])
        
    def calculate_structural_similarity(self, xml_node1: Dict, xml_node2: Dict) -> float:
        """Calculate structural similarity for XML nodes"""
        if not (xml_node1 and xml_node2):
            return 0.0
            
        # Compare node attributes and structure
        similarity_scores = []
        
        # Compare NodeIds
        if xml_node1.get("NodeId") == xml_node2.get("NodeId"):
            similarity_scores.append(1.0)
        
        # Compare References
        refs1 = set(str(ref) for ref in xml_node1.get("References", []))
        refs2 = set(str(ref) for ref in xml_node2.get("References", []))
        if refs1 or refs2:
            ref_similarity = len(refs1.intersection(refs2)) / len(refs1.union(refs2))
            similarity_scores.append(ref_similarity)
            
        # Compare other attributes
        for attr in ["DisplayName", "Description", "Value"]:
            if xml_node1.get(attr) and xml_node2.get(attr):
                attr_similarity = self.calculate_semantic_similarity(
                    str(xml_node1[attr]), str(xml_node2[attr])
                )
                similarity_scores.append(attr_similarity)
                
        return np.mean(similarity_scores) if similarity_scores else 0.0
        
    def calculate_ndcg(self, relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not relevance_scores:
            return 0.0
            
        dcg = 0
        idcg = 0
        sorted_relevance = sorted(relevance_scores, reverse=True)
        
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / np.log2(i + 2)
            idcg += sorted_relevance[i] / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
        
    def calculate_mrr(self, relevance_scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, score in enumerate(relevance_scores, 1):
            if score >= 0.5:  # Threshold for considering a result relevant
                return 1.0 / i
        return 0.0
        
    def evaluate_response_quality(self, response: str, true_label: str) -> float:
        """Evaluate the quality of the RAG response"""
        # Combine multiple metrics for response quality
        semantic_sim = self.calculate_semantic_similarity(response, true_label)
        bm25_score = self.calculate_bm25_score(true_label, response)
        
        # Normalize BM25 score to [0,1] range
        normalized_bm25 = np.tanh(bm25_score)  # Using tanh for smooth normalization
        
        # Combine scores with weights
        weights = {'semantic': 0.7, 'bm25': 0.3}
        quality_score = (
            weights['semantic'] * semantic_sim +
            weights['bm25'] * normalized_bm25
        )
        
        return float(quality_score)
        
    def evaluate_query_results(
        self,
        query_results: List[Dict],
        true_label: str,
        query: str,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[int, EvaluationResult]:
        """Evaluate query results against true label for different k values"""
        evaluation_results = {}
        
        for k in k_values:
            top_k_results = query_results[:k]
            detailed_metrics = []
            relevance_scores = []
            
            for result in top_k_results:
                # Calculate various similarity metrics
                content = result.get('content', '')
                response = result.get('rag_response', '')
                
                semantic_sim = self.calculate_semantic_similarity(content, true_label)
                bm25_score = self.calculate_bm25_score(query, content)
                
                # Calculate structural similarity for XML nodes
                structural_sim = None
                if result.get('metadata', {}).get('source_type') == 'xml':
                    structural_sim = self.calculate_structural_similarity(
                        result.get('original_details', {}),
                        {'Description': true_label}  # Simplified comparison
                    )
                
                # Calculate response quality if RAG response exists
                response_quality = (
                    self.evaluate_response_quality(response, true_label)
                    if response else 0.0
                )
                
                # Calculate combined relevance score
                weights = {
                    'semantic': 0.4,
                    'bm25': 0.3,
                    'structural': 0.1,
                    'response': 0.2
                }
                
                relevance_score = (
                    weights['semantic'] * semantic_sim +
                    weights['bm25'] * bm25_score +
                    weights['structural'] * (structural_sim or 0.0) +
                    weights['response'] * response_quality
                )
                
                relevance_scores.append(relevance_score)
                
                # Store detailed metrics
                detailed_metrics.append(DetailedMetrics(
                    semantic_similarity=semantic_sim,
                    bm25_score=bm25_score,
                    structural_similarity=structural_sim,
                    relevance_score=relevance_score,
                    response_quality=response_quality,
                    combined_score=relevance_score
                ))
            
            # Calculate traditional metrics
            relevant_results = sum(1 for score in relevance_scores if score >= 0.5)
            precision = relevant_results / k if k > 0 else 0
            recall = relevant_results / 1  # Assuming one true label
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate NDCG and MRR
            ndcg = self.calculate_ndcg(relevance_scores, k)
            mrr = self.calculate_mrr(relevance_scores)
            
            evaluation_results[k] = EvaluationResult(
                k=k,
                precision=precision,
                recall=recall,
                f1_score=f1,
                ndcg=ndcg,
                mrr=mrr,
                detailed_metrics=detailed_metrics
            )
            
        return evaluation_results

def print_evaluation_summary(evaluation_results: Dict[int, List[EvaluationResult]]):
    """Print comprehensive evaluation summary"""
    print("\nEnhanced Evaluation Summary:")
    print("=" * 70)
    
    for k, results in evaluation_results.items():
        print(f"\nTop-{k} Results:")
        print("-" * 70)
        
        # Calculate averages for all metrics
        avg_metrics = defaultdict(list)
        for result in results:
            avg_metrics['precision'].append(result.precision)
            avg_metrics['recall'].append(result.recall)
            avg_metrics['f1'].append(result.f1_score)
            avg_metrics['ndcg'].append(result.ndcg)
            avg_metrics['mrr'].append(result.mrr)
            
            # Detailed metrics
            for detailed in result.detailed_metrics:
                avg_metrics['semantic_sim'].append(detailed.semantic_similarity)
                avg_metrics['bm25'].append(detailed.bm25_score)
                if detailed.structural_similarity is not None:
                    avg_metrics['structural'].append(detailed.structural_similarity)
                avg_metrics['response_quality'].append(detailed.response_quality)
                avg_metrics['combined'].append(detailed.combined_score)
        
        # Print averaged results
        print(f"Traditional Metrics:")
        print(f"  Precision: {np.mean(avg_metrics['precision']):.4f}")
        print(f"  Recall: {np.mean(avg_metrics['recall']):.4f}")
        print(f"  F1 Score: {np.mean(avg_metrics['f1']):.4f}")
        print(f"  NDCG: {np.mean(avg_metrics['ndcg']):.4f}")
        print(f"  MRR: {np.mean(avg_metrics['mrr']):.4f}")
        
        print("\nDetailed Metrics:")
        print(f"  Semantic Similarity: {np.mean(avg_metrics['semantic_sim']):.4f}")
        print(f"  BM25 Score: {np.mean(avg_metrics['bm25']):.4f}")
        if avg_metrics['structural']:
            print(f"  Structural Similarity: {np.mean(avg_metrics['structural']):.4f}")
        print(f"  Response Quality: {np.mean(avg_metrics['response_quality']):.4f}")
        print(f"  Combined Score: {np.mean(avg_metrics['combined']):.4f}")

def get_cache_path(file_path: str) -> Path:
    """
    Generate a unique cache file path based on the input file path and its content hash
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate hash of file content to ensure cache invalidation if file changes
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # Create cache filename using original filename and content hash
    original_filename = Path(file_path).stem
    cache_filename = f"{original_filename}_{file_hash}.pickle"
    
    return cache_dir / cache_filename

def save_to_cache(file_path: str, collection, raw_nodes: Dict):
    """
    Save processed data to cache
    """
    cache_path = get_cache_path(file_path)
    try:
        # Convert ChromaDB collection to serializable format
        collection_data = {
            'documents': collection.get()
        }
        
        cache_data = {
            'collection_data': collection_data,
            'raw_nodes': raw_nodes
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved to {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {str(e)}")

def load_from_cache(file_path: str) -> Tuple[Optional[Any], Optional[Dict]]:
    """
    Load processed data from cache if available
    """
    cache_path = get_cache_path(file_path)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Recreate ChromaDB collection from cached data
            client = chromadb.Client()
            collection = client.create_collection(
                name="document_embeddings",
                get_or_create=True
            )
            
            # Restore collection data
            collection_data = cache_data['collection_data']
            if collection_data['documents']:
                collection.add(
                    documents=collection_data['documents']['documents'],
                    metadatas=collection_data['documents']['metadatas'],
                    ids=collection_data['documents']['ids']
                )
            
            print("Loaded data from cache")
            return collection, cache_data['raw_nodes']
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return None, None
    return None, None

def process_file_with_cache(file_path: str) -> Tuple[Any, Dict]:
    """
    Process file with caching functionality
    """
    # Try to load from cache first
    collection, raw_nodes = load_from_cache(file_path)
    
    if collection is not None and raw_nodes is not None:
        return collection, raw_nodes
    
    # If no cache available, process the file
    print("No cache found or cache invalid. Processing file...")
    collection, raw_nodes = process_file(file_path)
    
    # Save to cache for future use
    if collection is not None and raw_nodes is not None:
        save_to_cache(file_path, collection, raw_nodes)
    
    return collection, raw_nodes


def main():
    # Initialize evaluator
    evaluator = EnhancedEvaluator()
    
    # Load and process document
    file_path = "./OPC2.xml"
    print("\nProcessing document...")
    collection, raw_nodes = process_file_with_cache(file_path)
    
    # Load test dataset
    dataset_path = "./queries_dataset.csv"
    test_data = load_test_dataset(dataset_path)
    
    if test_data:
        print("\nRunning enhanced evaluation...")
        all_results = defaultdict(list)
        
        for query, true_label in test_data:
            # Get system results
            results, _ = query_documents(collection, raw_nodes, query)
            
            # Evaluate results
            evaluation = evaluator.evaluate_query_results(results, true_label, query)
            
            # Store results
            for k, result in evaluation.items():
                all_results[k].append(result)
        
        # Print comprehensive results
        print_evaluation_summary(all_results)
    else:
        print("No valid test cases found in the dataset.")

if __name__ == "__main__":
    main()
