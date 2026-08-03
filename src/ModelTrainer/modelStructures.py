import ModelTrainer.graphFunctions as graphFunctions
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool
import ast

def load_model(path, model):
    model_params = torch.load(path, weights_only=True)
    model.load_state_dict(model_params['model_state_dict'])

# Container for easier accesss of the data 
class NodeTextData:
    def __init__(self, pyg_data, node_words: list[str], question: str, chunks: list[str]):
        self.pyg_data   = pyg_data      # torch_geometric.data.Data  (x is spaCy placeholder)
        self.node_words = node_words    # list[str], one entry per graph node
        self.question   = question
        self.chunks     = chunks

# Generates the graph structure, collecting and creating multiple NodeTextData objects 
class GraphDataset(Dataset):
    def __init__(self, tuples: list, cache=True): # Might turn the cache off if memory can't store it
        self.tuples = tuples
        self.cache = cache
        self._structure_cache = {} if cache else None

    def __len__(self):
        return len(self.tuples)

    def _build_structure(self, idx):
        question, chunks, label = self.tuples[idx]
        pyg_graph, node_words = graphFunctions.process_instance(question, chunks)
        pyg_graph.y = torch.tensor([label], dtype=torch.long)
        return NodeTextData(pyg_graph, node_words, question, chunks)

    def __getitem__(self, idx):
        if self.cache: # If using cache
            if idx not in self._structure_cache: # Build the graph and save in cache in the first iteration, only returning the cached structure after that
                self._structure_cache[idx] = self._build_structure(idx)
            return self._structure_cache[idx]
        return self._build_structure(idx) # If not, always build the structure

# Converts the dataframe loaded from a csv into a tuple containing the question, a list of context chunks, and the label/target value 
def convert_to_tuple(df):
    question_context_tuples = []
    for row in df.iterrows():
        row_data = row[1]
        question = row_data['question']
        label = row_data['label']
        chunks = ast.literal_eval(row_data['context'])
        chunks_list = []
        for chunk in chunks:
            chunks_list.append(chunk)

        question_context_tuples.append( (question, chunks_list, label) )

    return question_context_tuples

# Process the text and generate embeddings to them, associating the embeddings with the graph nodes
class BertNodeEmbedder(nn.Module):
    def __init__(self, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)

    def forward(self, node_words: list[str], question: str, chunks: list[str]) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        MAX_LEN = self.bert.config.max_position_embeddings # Maximum tokens for the model
        H = self.bert.config.hidden_size # Hidden size

        enc_query  = self.tokenizer(question, return_tensors="pt", truncation=False) # Tokenize query without truncating to the maximum
        query_ids  = enc_query["input_ids"][0][1:-1] # strip CLS and final SEP

        # max_query_len = MAX_LEN // 2 # Leave at least half the context for the 
        # if len(query_ids) > max_query_len:
        #     query_ids = query_ids[:max_query_len]

        chunk_budget = MAX_LEN - 3 - len(query_ids) # The chunk might have up to chunk_budget tokens, considering that CLS, SEP (between query and context) and SEP (in the end) and the tokens used for the query

        # Get SEP and CLS tokens
        sep_id = torch.tensor([self.tokenizer.sep_token_id])
        cls_id = torch.tensor([self.tokenizer.cls_token_id])
        # query_tail_ids = torch.cat([query_ids, sep_id])

        # Tokenize the full chunk text, keeping character offsets to match subword tokens with word tokens later
        full_text = " ".join(chunks)
        enc_chunk = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False
        )
        all_chunk_ids = enc_chunk["input_ids"][0][1:-1] # strip CLS and final SEP
        all_offsets = enc_chunk["offset_mapping"][0][1:-1] # strip CLS and final SEP
        num_chunk_tokens = len(all_chunk_ids)

        # Generate windows of chunks, accumulating hidden states per token position
        # position_hidden: chunk token index → list of hidden vectors
        position_hidden: dict[int, list[torch.Tensor]] = {}

        step = max(1, chunk_budget // 2) # How much the window slide before encoding the next window 
        window_start = 0

        while window_start < num_chunk_tokens:
            window_end = min(window_start + chunk_budget, num_chunk_tokens) # End the window considering the maximum context length, with caution not to end within the actual text
            window_ids = all_chunk_ids[window_start:window_end] # Get only the tokens in this window

            # Create the text processed by BERT: [CLS] query [SEP] chunk_window [SEP]
            input_ids = torch.cat([
                cls_id,
                query_ids,
                sep_id,
                window_ids,
                sep_id
            ]).unsqueeze(0).to(device)

            attention_mask  = torch.ones_like(input_ids) # No padding
            type_0_len      = 1 + len(query_ids) + 1 # CLS + query + SEP
            type_1_len      = len(window_ids) + 1 # chunk + SEP
            token_type_ids  = torch.cat([
                torch.zeros(type_0_len, dtype=torch.long),
                torch.ones(type_1_len, dtype=torch.long)
            ]).unsqueeze(0).to(device) # Type ids inform that those are two different sentences

            hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).last_hidden_state[0]   # [seq_len, H]

    
            # Add the embedding to the list of embeddings for that token (global_i), using the local position to index it
            for local_i, global_i in enumerate(range(window_start, window_end)):
                position_hidden.setdefault(global_i, []).append(hidden[local_i + type_0_len]) # type_0_len represents CLS + query + SEP, marking the start of the chunk

            if window_end == num_chunk_tokens:
                break
            window_start += step # Move the window

        # Average the tokens embeddings across windows, resulting in one vector per token position
        position_embedding: dict[int, torch.Tensor] = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in position_hidden.items()
        }

        
        doc = graphFunctions.nlp(full_text) # Build spaCy doc to map subwords to words
        token_vecs: dict[str, list[torch.Tensor]] = {}

        for global_i, (char_start, char_end) in enumerate(all_offsets.tolist()):
            if char_start == char_end: # If padding or special token
                continue
            if global_i not in position_embedding: # If subword token has no embedding
                continue
            for spacy_tok in doc: # For each work token 
                tok_start = spacy_tok.idx
                tok_end   = spacy_tok.idx + len(spacy_tok.text)
                # If subword is within word, add the embedding to the list of embeddings of that word
                if char_start >= tok_start and char_end <= tok_end: 
                    token_vecs.setdefault(spacy_tok.text.lower(), []).append(position_embedding[global_i])
                    break

        # Average the subword emebddings within the same word, resulting in word embeddings
        word_to_embedding: dict[str, torch.Tensor] = {
            word: torch.stack(vecs).mean(dim=0)
            for word, vecs in token_vecs.items()
        }

        # Reorder the embeddings to the node_words order
        embeddings = []
        for word in node_words:
            if word in word_to_embedding:
                embeddings.append(word_to_embedding[word])
            else:
                # Fallback for any errors, should never happen
                all_vecs = list(word_to_embedding.values())
                if all_vecs:
                    embeddings.append(torch.stack(all_vecs).mean(dim=0))
                else:
                    embeddings.append(torch.zeros(H, device=device))

        return torch.stack(embeddings) # [N, H]

class GATWithBERT(nn.Module):
    def __init__(
        self,
        bert_model_name:  str  = "bert-base-uncased",
        hidden_channels:  int  = 128,
        num_classes:      int  = 2,
        heads:            int  = 4,
        dropout_rate:     float = 0.1,
        freeze_bert_layers: int = 8,    # Freeze first N transformer layers
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.embedder = BertNodeEmbedder(bert_model_name)
        in_channels   = self.embedder.bert.config.hidden_size # 768 for default BERT

        self._freeze_bert_layers(freeze_bert_layers) # Freeze BERT layers to avoid catastrophic forget and increase training performance

        # GAT Layers
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout_rate)

        # Classification head
        mlp_input_dim = (hidden_channels * heads) + hidden_channels
        mlp_hidden    = hidden_channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden, num_classes)
        )

    def _freeze_bert_layers(self, n: int):
        # Freeze the look-up table, for changing it could be forcing the embeddings to represent what's useful for the task instead of what the word means, introducing shortcut learning
        for param in self.embedder.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first N transformer layers from BERT, leaving the last Layers - N layers to be fine-tuned    
        for layer in self.embedder.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def _get_number_of_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"Total params": total_params, "Trainable params": trainable_params}

    def forward(self, batch: list):
        device = next(self.parameters()).device
        all_gat_logits = []

        for item in batch:
            pyg   = item.pyg_data
            words = item.node_words

            # Embed the chunks considering the question, ordering with the same index as words
            x = self.embedder(words, item.question, item.chunks) # [words, dim], in BERT, [words, 768]

            edge_index = pyg.edge_index.to(device)
            batch_vec  = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            # GAT processing (shape examples considering 8 heads of hidden_dim = 32)
            x = self.conv1(x, edge_index) # [words, 256]
            x = F.elu(x) # Non-linear activation function
            x = F.dropout(x, p=self.dropout_rate, training=self.training) # Regularization
            # Skip connection, keeping the first GAT knowledge and helping in regularization. Mean pool transforms from [words, 256] to [BATCH, 256], one embedding per batch
            skip = global_mean_pool(x, batch_vec) # [BATCH, 256]

            x = self.conv2(x, edge_index) # [words, 32]
            x = F.elu(x) # Non-linear activation function
            x = F.dropout(x, p=self.dropout_rate, training=self.training) # Regularization
            x = global_mean_pool(x, batch_vec) # [BATCH, 32]

            # Concatenate [BATCH, 256] with [BATCH, 32]
            x = torch.cat([skip, x], dim=1) # [BATCH, 288]
            all_gat_logits.append(x)

        logits = self.mlp(torch.cat(all_gat_logits, dim=0)) # [BATCH, 288] -> [BATCH, 64] -> [BATCH, 2], the final prediction
        return logits