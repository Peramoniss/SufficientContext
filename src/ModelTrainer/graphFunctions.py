import gc
import spacy
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
import random

BERT_MODEL = "bert-base-uncased" # Define hugging face's BERT model address

# Set the seed in every randomness-dependent library 
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Change BERT model dynamically, if wanted
def set_bert(new_bert_model: str):
    global BERT_MODEL
    BERT_MODEL = new_bert_model

# Clean GPU and RAM memory  
def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

nlp = spacy.load("en_core_web_lg", disable=["ner"]) # Dependency parser; syntactic knowledge 
tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL)
bert_model.eval() # Fix BERT into evaluation mode until training occurs

def graphy(text):
  doc = nlp(text) # Parse the text
  G = nx.DiGraph() # Initialize a Directed Graph using NetworkX

  # Use the dependency analysis to build the graph
  for token in doc: # For each word (spacy token)
      G.add_edge(token.head.text.lower(), token.text.lower(), label=token.dep_) # Add an edge from HEAD to the word, using the dependency label as the edge attribute

  return G, doc

def convert_nx_to_pyg(G): # Convert nx to pyg
    # Indexes the words that represent the nodes
    nodes_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}

    # Convert edges format. PyG expects a shape of [2, num_edges] containing source and target indices
    edge_indices = []
    # edge_labels = []
    for u, v, data in G.edges(data=True):
        source_idx = node_to_idx[u]
        target_idx = node_to_idx[v]
        edge_indices.append([source_idx, target_idx])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() # Transpose to get the required [2, num_edges] shape

    pyg_data = Data(edge_index=edge_index) # Finally converts into pyg

    return pyg_data, node_to_idx

# Define nodes and edges to filter
stop_words = nlp.Defaults.stop_words
keep_words = set([
    # Negations
    "not", "no", "never", "none", "neither", "nor", "without", "against",
    # Interrogatives
    "what", "who", "where", "when", "why", "how", "which", "whose",
    # Modals / Intent
    "must", "should", "could", "would", "will", "can", "might", "may",
    # Personal Pronouns
    "i", "me", "my", "myself", "you", "your", "he", "him", "his", "she",
    "her", "we", "us", "our", "they", "them", "their",
    # Temporal
    "before", "after", "above", "below", "between", "during", "under", "over"
])

remove_punct = [
    "!", ".", ",", "/", "\\", "[", "]", "{", "}", ";", ":", "&", "#", "(", ")"
    # "?", ">", "<", "*", "-", "$" are exceptions
]

stopwords_to_clean = stop_words - set(keep_words)
stopwords_to_clean.update(remove_punct)
unwanted_edges = {'punct', 'det', 'dep'}

def filter_dependency_graph(G, structural_stopwords):
    filtered_G = G.copy() # Work on a copy so we don't mutate the original graph unexpectedly

    # Remove unwanted edge types
    edges_to_remove = [
        (u, v) for u, v, data in filtered_G.edges(data=True)
        if data.get('label') in unwanted_edges
    ]
    filtered_G.remove_edges_from(edges_to_remove)

    # Remove stopword nodes
    nodes_to_process = list(filtered_G.nodes())
    for node in nodes_to_process:
        if node.lower() in structural_stopwords: # Check if the lowercase version of the node text is a stopword
            # In NetworkX DiGraph, predecessors are incoming (parents), successors are outgoing (children)
            parents = list(filtered_G.predecessors(node))
            children = list(filtered_G.successors(node))

            # Link parents directly to children to preserve path continuity
            for parent in parents:
                for child in children:
                    # Inherit the relationship label from the parent-to-stopword connection
                    edge_label = filtered_G.edges[parent, node].get('label', 'mod')
                    filtered_G.add_edge(parent, child, label=edge_label)

            filtered_G.remove_node(node)

    # Remove isolated nodes after the edge removal
    nodes_to_remove = [
        node for node, _ in filtered_G.nodes(data=True)
        if filtered_G.degree(node) == 0
    ]

    filtered_G.remove_nodes_from(nodes_to_remove)

    return filtered_G

def process_instance(question: str, chunks: list):
    iter = [question] + chunks

    complete_G = None
    for chunk in iter:
        G, _ = graphy(chunk)
        complete_G = G if complete_G is None else nx.compose(complete_G, G)

    complete_G = filter_dependency_graph(complete_G, stopwords_to_clean)

    pyg_graph, node_to_id = convert_nx_to_pyg(complete_G)

    id_to_node = {v: k for k, v in node_to_id.items()} # Invert the node to id dictionary
    node_words  = [id_to_node[i] for i in range(len(id_to_node))] # Convert to a list

    return pyg_graph, node_words