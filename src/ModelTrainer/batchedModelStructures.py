import ModelTrainer.graphFunctions as graphFunctions
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
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
    def __init__(self, bert_model_name: str = "bert-base-uncased", bert_sub_batch: int = 16):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_sub_batch = bert_sub_batch

    # node_words: list[str], question: str, chunks: list[str]
    def forward(self, batch: list) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        MAX_LEN = self.bert.config.max_position_embeddings # Maximum tokens for the model
        H = self.bert.config.hidden_size # Hidden size

        # Get SEP and CLS tokens
        sep_id = torch.tensor([self.tokenizer.sep_token_id])
        cls_id = torch.tensor([self.tokenizer.cls_token_id])
        pad_id = self.tokenizer.pad_token_id
        batch_data = []
        windows_data = []
        windows_indexer = []

        for i, item in enumerate(batch):
            enc_query  = self.tokenizer(item.question, return_tensors="pt", truncation=False) # Tokenize query without truncating to the maximum
            query_ids  = enc_query["input_ids"][0][1:-1] # strip CLS and final SEP

            chunk_budget = MAX_LEN - 3 - len(query_ids) # The chunk might have up to chunk_budget tokens, considering that CLS, SEP (between query and context) and SEP (in the end) and the tokens used for the query

            # Tokenize the full chunk text, keeping character offsets to match subword tokens with word tokens later
            full_text = " ".join(item.chunks)

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
            # position_hidden: dict[int, list[torch.Tensor]] = {}

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
                ])

                attention_mask  = torch.ones_like(input_ids) # No padding
                type_0_len      = 1 + len(query_ids) + 1 # CLS + query + SEP
                type_1_len      = len(window_ids) + 1 # chunk + SEP

                windows_data.append({
                    "input_ids": input_ids, 
                    "type_0_len": type_0_len, 
                    "type_1_len": type_1_len,
                    "pos_on_batch": i,
                    "window_id_within_batch": len(windows_indexer), # In this instance within the batch, this window represents the X positioned window of the batch
                })
                windows_indexer.append((window_start, window_end))

                # if window_end == num_chunk_tokens:
                #     break
                window_start += step # Move the window

            batch_data.append(
                {
                    "full_text": full_text,
                    "all_offsets": all_offsets,
                    "num_chunk_tokens": num_chunk_tokens,
                    "windows_indexer": windows_indexer,
                    "position_hidden": {},  # filled later
                }
            )

        # Sub-batch needed because a single instance might become several 512-token inputs, so the user might think the GPU handles a 32-instance batch, but in reality it is a 128-instance for BERT, for example
        for starting_window in range(0, len(windows_data), self.bert_sub_batch):
            sub_batch = windows_data[starting_window:starting_window+self.bert_sub_batch]
            max_len = max(len(w["input_ids"]) for w in sub_batch) # Find the biggest length within the batch to pad the remaining (most windows will be locked in the maximum size, but tail batches might have this issue)

            input_ids_batch = torch.full((len(sub_batch), max_len), pad_id, dtype=torch.long) # Start the instance with all padding
            attention_batch = torch.zeros((len(sub_batch), max_len), dtype=torch.long) # Start attention considering everything is padding
            token_type_batch = torch.zeros((len(sub_batch), max_len), dtype=torch.long) # Start token_type considering everything is the same sentence

            for i, curr_window in enumerate(sub_batch):
                window_length = len(curr_window["input_ids"])
                input_ids_batch[i, :window_length] = curr_window["input_ids"] # Overwrite padding to become the actual text, leaving what's not in the window_length as padding
                attention_batch[i, :window_length] = 1 # Sets everything within window_length as attention-needed (non-padding)
                token_type_batch[i, curr_window["type_0_len"]:window_length] = 1  # Marks the chunk as a difference sentence than the question

            hidden = self.bert(
                input_ids=input_ids_batch.to(device),
                attention_mask=attention_batch.to(device),
                token_type_ids=token_type_batch.to(device)
            ).last_hidden_state  # [sub_batch, max_len, H]

            for i, window in enumerate(sub_batch):
                pos_on_batch = window["pos_on_batch"] # Discover from which instance of the batch it belongs to
                item_windows_data = batch_data[pos_on_batch]["windows_indexer"] # Get the data from windows for that instance of the batch
                window_start, window_end = item_windows_data[window["window_id_within_batch"]] # Get where in the full text this specific window is 
                type_0_len = window["type_0_len"]
                pos_hidden = batch_data[pos_on_batch]["position_hidden"]
                # Add the embedding of each token to the list of embeddings for that token (original_text_i), using the local position to index it
                for local_i, original_text_i in enumerate(range(window_start, window_end)):
                    pos_hidden.setdefault(original_text_i, []).append(hidden[i, local_i + type_0_len])


        results = []
        for i, item in enumerate(batch):
            data = batch_data[i]

            # Average the tokens embeddings across windows, resulting in one vector per token position
            position_embedding: dict[int, torch.Tensor] = {
                pos: torch.stack(vecs).mean(dim=0)
                for pos, vecs in data["position_hidden"].items()
            }

            doc = graphFunctions.nlp(data["full_text"]) # Build spaCy doc to map subwords to words
            token_vecs: dict[str, list[torch.Tensor]] = {}

            for global_i, (char_start, char_end) in enumerate(data["all_offsets"].tolist()):
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
            for word in item.node_words:
                if word in word_to_embedding:
                    embeddings.append(word_to_embedding[word])
                else:
                    # Fallback for any errors
                    all_vecs = list(word_to_embedding.values())
                    if all_vecs:
                        embeddings.append(torch.stack(all_vecs).mean(dim=0))
                    else:
                        embeddings.append(torch.zeros(H, device=device))
            results.append(torch.stack(embeddings)) # [N, H]

        return results # [BATCH, N, H]

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
        embeddings = self.embedder(batch) # [BATCH, words, dim], in BERT, [BATCH, words, 768]

        # Convert embeddings in PyTorchGeometric structure
        data_list = []
        for item, x in zip(batch, embeddings):
            d = item.pyg_data.clone()
            d.x = x
            data_list.append(d)

        graph_batch = Batch.from_data_list(data_list).to(device)
        x, batch_edge_index, batch_vec = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        all_gat_logits = []

        # Embed the chunks considering the question, ordering with the same index as words
        # GAT processing (shape examples considering 8 heads of hidden_dim = 32)
        x = self.conv1(x, batch_edge_index) # [words, 256]
        x = F.elu(x) # Non-linear activation function
        x = F.dropout(x, p=self.dropout_rate, training=self.training) # Regularization
        # Skip connection, keeping the first GAT knowledge and helping in regularization. Mean pool transforms from [words, 256] to [BATCH, 256], one embedding per batch
        skip = global_mean_pool(x, batch_vec) # [BATCH, 256]

        x = self.conv2(x, batch_edge_index) # [words, 32]
        x = F.elu(x) # Non-linear activation function
        x = F.dropout(x, p=self.dropout_rate, training=self.training) # Regularization
        x = global_mean_pool(x, batch_vec) # [BATCH, 32]

        # Concatenate [BATCH, 256] with [BATCH, 32]
        x = torch.cat([skip, x], dim=1) # [BATCH, 288]
        
        logits = self.mlp(x) # [BATCH, 288] -> [BATCH, 64] -> [BATCH, 2], the final prediction
        return logits