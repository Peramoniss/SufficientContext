import spacy
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
import random
from sklearn.feature_extraction.text import TfidfVectorizer

BERT_MODEL = "bert-base-uncased" # Define hugging face's BERT model address
SEED = 0

# Set the seed in every randomness-dependent library 
def set_seed(seed):
    global SEED
    SEED = seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_seed():
    return SEED

# Change BERT model dynamically, if wanted
def set_bert(new_bert_model: str):
    global BERT_MODEL
    BERT_MODEL = new_bert_model

# Define nodes and edges to filter
nlp = spacy.load("en_core_web_lg", disable=["ner"]) # Dependency parser; syntactic knowledge 
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
tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL)
# bert_model = AutoModel.from_pretrained(BERT_MODEL)
# bert_model.eval() # Fix BERT into evaluation mode until training occurs

def graphy(text, semantic = True):
    doc = nlp(text) # Parse the text
    G = nx.DiGraph() # Initialize a Directed Graph using NetworkX

    # Use the dependency analysis to build the graph
    for token in doc: # For each word (spacy token)
        G.add_edge(token.head.text.lower(), token.text.lower(), label=token.dep_) # Add an edge from HEAD to the word, using the dependency label as the edge attribute

    if not semantic:
        # Attach word2vec-style embeddings as node features when the graph is syntactic-only
        for node in G.nodes():
            lex = nlp.vocab[node]  # look up by string, no need to re-parse
            if lex.has_vector:
                G.nodes[node]["embedding"] = torch.tensor(lex.vector, dtype=torch.float)
            else:
                G.nodes[node]["embedding"] = torch.zeros(nlp.vocab.vectors_length, dtype=torch.float)

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

    if "embedding" in G.nodes[nodes_list[0]]: # Converts to pyg with fixed embeddings
        x = torch.stack([G.nodes[node]["embedding"] for node in nodes_list])
        pyg_data = Data(x=x, edge_index=edge_index)
    else:
        pyg_data = Data(edge_index=edge_index) # Finally converts into pyg

    return pyg_data, node_to_idx

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

def precompute_bert_alignment(chunks: list):
    """
    Precomputes everything BertNodeEmbedder needs that only depends on
    `chunks`, not on model weights or the training epoch:
      - BERT token ids for the joined chunk text (no CLS/SEP, no truncation)
      - Their character offsets
      - spaCy word boundaries (start, end, lowercased text) for the same text

    This used to be redone from scratch inside BertNodeEmbedder.forward on
    every single forward pass. Since it only depends on `chunks`, it's safe
    to compute once here and cache it on the Dataset item instead.
    """
    full_text = " ".join(chunks)

    enc_chunk = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=False
    )
    chunk_ids = enc_chunk["input_ids"][0][1:-1]        # strip CLS and final SEP
    chunk_offsets = enc_chunk["offset_mapping"][0][1:-1]

    doc = nlp(full_text)
    # spaCy tokens are already left-to-right, so this list is sorted for free
    spacy_bounds = [
        (tok.idx, tok.idx + len(tok.text), tok.text.lower())
        for tok in doc
    ]

    return chunk_ids, chunk_offsets, spacy_bounds

def process_semantic_instance(question: str, chunks: list):
    iterr = [question] + chunks

    complete_G = None
    for chunk in iterr:
        G, _ = graphy(chunk, semantic=True)
        complete_G = G if complete_G is None else nx.compose(complete_G, G)

    complete_G = filter_dependency_graph(complete_G, stopwords_to_clean)

    pyg_graph, node_to_id = convert_nx_to_pyg(complete_G)

    id_to_node = {v: k for k, v in node_to_id.items()}
    node_words  = [id_to_node[i] for i in range(len(id_to_node))]

    chunk_ids, chunk_offsets, spacy_bounds = precompute_bert_alignment(chunks)

    return pyg_graph, node_words, chunk_ids, chunk_offsets, spacy_bounds

def process_doc_graph_instance(chunks: list[str], threshold = 0.15):
    num_chunks = len(chunks)
    # 1. Compute term frequency/overlap using CountVectorizer or TfidfVectorizer
    # This automatically handles lowercasing, tokenization, and stop words.
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    # Compute dot product to get pairwise overlap / similarity
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    sources, targets = [], []
    #Complete graph (every document with every document)
    for i in range(num_chunks):
        for j in range(num_chunks):
            if i != j and similarity_matrix[i, j] > 0.15:
                sources.append(i)
                targets.append(j)
    
    # # 3. Extract upper triangle without the diagonal (i < j)
    # upper_i, upper_j = np.triu_indices_from(similarity_matrix, k=1)
    # mask = similarity_matrix[upper_i, upper_j] >= threshold
    # filtered_sources = upper_i[mask]
    # filtered_targets = upper_j[mask]

    # # # 5. Mirror edges to make the graph bidirectional (i -> j and j -> i)
    # sources = np.concatenate([filtered_sources, filtered_targets])
    # targets = np.concatenate([filtered_targets, filtered_sources])
                
    # edge_indexes = torch.tensor([sources, targets], dtype=torch.long)
    edge_indexes = torch.from_numpy(np.stack([sources, targets])).long()
    pyg_graph = Data(edge_index=edge_indexes)

    chunks_ids = []
    for chunk in chunks:
        enc_chunk = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=False
        )
        chunks_ids.append(enc_chunk["input_ids"][0])

    chunks_ids = torch.nn.utils.rnn.pad_sequence(chunks_ids, batch_first=True, padding_value=tokenizer.pad_token_id)       
    return pyg_graph, chunks_ids

def process_syntactic_only_instance(question: str, chunks: list):
    iterr = [question] + chunks

    complete_G = None
    for chunk in iterr:
        G, _ = graphy(chunk, semantic=False)
        complete_G = G if complete_G is None else nx.compose(complete_G, G)

    complete_G = filter_dependency_graph(complete_G, stopwords_to_clean)

    pyg_graph, node_to_id = convert_nx_to_pyg(complete_G)

    id_to_node = {v: k for k, v in node_to_id.items()}
    node_words  = [id_to_node[i] for i in range(len(id_to_node))]

    return pyg_graph, node_words # Maybe not even node_words