from typing import override

import ModelTrainer.graphFunctions as graphFunctions
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import ast

def load_model(path, model):
    model_params = torch.load(path, weights_only=True)
    model.load_state_dict(model_params['model_state_dict'])

# Container for easier accesss of the data 
class NodeTextData:
    def __init__(self, pyg_data, node_words: list[str], question: str, chunks: list[str], chunk_ids: torch.Tensor = None, chunk_offsets: torch.Tensor = None, spacy_bounds: list[tuple] = None):
        self.pyg_data   = pyg_data      # torch_geometric.data.Data  (x is spaCy placeholder)
        self.y = pyg_data.y
        self.node_words = node_words    # list[str], one entry per graph node
        self.question   = question
        self.chunks     = chunks
        self.chunk_ids     = chunk_ids       # precomputed BERT ids for joined chunks
        self.chunk_offsets = chunk_offsets   # precomputed char offsets, aligned to chunk_ids
        self.spacy_bounds  = spacy_bounds    # precomputed [(start, end, word_lower), ...]

# Generates the graph structure, collecting and creating multiple NodeTextData objects 
class GraphDataset(Dataset):
    def __init__(self, tuples: list, cache=False): # Might turn the cache off if memory can't store it
        self.tuples = tuples
        self.cache = cache
        self._structure_cache = {} if cache else None

    def __len__(self):
        return len(self.tuples)

    def _build_structure(self, idx):
        question, chunks, label = self.tuples[idx]
        pyg_graph, node_words, chunk_ids, chunk_offsets, spacy_bounds = graphFunctions.process_semantic_instance(question, chunks)
        pyg_graph.y = torch.tensor([label], dtype=torch.long)
        return NodeTextData(pyg_graph, node_words, question, chunks, chunk_ids, chunk_offsets, spacy_bounds)

    # TODO: REMOVE CACHE
    def __getitem__(self, pos):
        if self.cache: # If using cache
            if pos not in self._structure_cache: # Build the graph and save in cache in the first iteration, only returning the cached structure after that
                self._structure_cache[pos] = self._build_structure(pos)
            return self._structure_cache[pos]
        return self._build_structure(pos) # If not, always build the structure

class SyntacticGraphDataset(GraphDataset):
    def __init__(self, tuples: list, cache=False):
        super().__init__(tuples, cache)

    @override
    def _build_structure(self, idx):
        question, chunks, label = self.tuples[idx]
        pyg_graph, node_words = graphFunctions.process_syntactic_only_instance(question, chunks)
        pyg_graph.y = torch.tensor([label], dtype=torch.long)
        return NodeTextData(pyg_graph, node_words, question, chunks)


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
class BertWordEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.tokenizer = graphFunctions.tokenizer
        self.bert = AutoModel.from_pretrained(graphFunctions.BERT_MODEL) # bert_model_name
        # self.bert = graphFunctions.bert_model

    def forward(self, node_words: list[str], question: str,
            chunk_ids: torch.Tensor, chunk_offsets: torch.Tensor,
            spacy_bounds: list[tuple]) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        MAX_LEN = self.bert.config.max_position_embeddings
        H = self.bert.config.hidden_size

        enc_query = self.tokenizer(question, return_tensors="pt", truncation=False)
        query_ids = enc_query["input_ids"][0][1:-1]

        chunk_budget = MAX_LEN - 3 - len(query_ids)

        sep_id = torch.tensor([self.tokenizer.sep_token_id])
        cls_id = torch.tensor([self.tokenizer.cls_token_id])

        all_chunk_ids = chunk_ids          # now precomputed, passed in
        all_offsets = chunk_offsets        # now precomputed, passed in
        num_chunk_tokens = len(all_chunk_ids)

        position_hidden: dict[int, list[torch.Tensor]] = {}

        # step = max(1, chunk_budget // 2)
        step = max(1, chunk_budget * 3 // 4) # Passos mais largos
        window_start = 0

        while window_start < num_chunk_tokens:
            window_end = min(window_start + chunk_budget, num_chunk_tokens)
            window_ids = all_chunk_ids[window_start:window_end]

            input_ids = torch.cat([
                cls_id, query_ids, sep_id, window_ids, sep_id
            ]).unsqueeze(0).to(device)

            attention_mask = torch.ones_like(input_ids)
            type_0_len = 1 + len(query_ids) + 1
            type_1_len = len(window_ids) + 1
            token_type_ids = torch.cat([
                torch.zeros(type_0_len, dtype=torch.long),
                torch.ones(type_1_len, dtype=torch.long)
            ]).unsqueeze(0).to(device)

            hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).last_hidden_state[0]

            for local_i, global_i in enumerate(range(window_start, window_end)):
                position_hidden.setdefault(global_i, []).append(hidden[local_i + type_0_len])

            if window_end == num_chunk_tokens:
                break
            window_start += step

        position_embedding: dict[int, torch.Tensor] = {
            pos: torch.stack(vecs).mean(dim=0)
            for pos, vecs in position_hidden.items()
        }

        # --- Fast pointer-based matching, no spaCy call, no rescans ---
        token_vecs: dict[str, list[torch.Tensor]] = {}
        num_spacy = len(spacy_bounds)
        tok_ptr = 0

        offsets_list = all_offsets.tolist()

        for global_i, (char_start, char_end) in enumerate(offsets_list):
            if char_start == char_end:
                continue
            if global_i not in position_embedding:
                continue

            while tok_ptr < num_spacy and spacy_bounds[tok_ptr][1] < char_start:
                tok_ptr += 1

            j = tok_ptr
            while j < num_spacy:
                tok_start, tok_end, tok_text = spacy_bounds[j]
                if tok_start > char_end:
                    break
                if char_start >= tok_start and char_end <= tok_end:
                    token_vecs.setdefault(tok_text, []).append(position_embedding[global_i])
                    break
                j += 1

        word_to_embedding: dict[str, torch.Tensor] = {
            word: torch.stack(vecs).mean(dim=0)
            for word, vecs in token_vecs.items()
        }

        embeddings = []
        for word in node_words:
            if word in word_to_embedding:
                embeddings.append(word_to_embedding[word])
            else:
                all_vecs = list(word_to_embedding.values())
                if all_vecs:
                    embeddings.append(torch.stack(all_vecs).mean(dim=0))
                else:
                    embeddings.append(torch.zeros(H, device=device))

        return torch.stack(embeddings)

class SyntacticGATWithBERT(nn.Module):
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

        self.embedder = BertWordEmbedder()
        in_channels   = self.embedder.bert.config.hidden_size # 768 for default BERT

        self.freeze_bert_layers = freeze_bert_layers
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
        data_list = []

        for item in batch:
            words = item.node_words

            # Embed the chunks considering the question, ordering with the same index as words
            # x = self.embedder(words, item.question, item.chunks) # [words, dim], in BERT, [words, 768]
            x = self.embedder(words, item.question, item.chunk_ids, item.chunk_offsets, item.spacy_bounds)
            data_list.append(Data(x=x, edge_index=item.pyg_data.edge_index))

        merged = Batch.from_data_list(data_list).to(device)
        x, edge_index, batch_vec = merged.x, merged.edge_index, merged.batch

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

        logits = self.mlp(torch.cat([skip, x], dim=1)) # [BATCH, 288] -> [BATCH, 64] -> [BATCH, 2], the final prediction
        return logits

class SyntacticGAT(nn.Module):
    def __init__(
        self,
        bert_model_name:  str  = "bert-base-uncased",
        hidden_channels:  int  = 128,
        num_classes:      int  = 2,
        heads:            int  = 4,
        dropout_rate:     float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        in_channels   = 300 # 300 for default word2vec

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

    def _get_number_of_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"Total params": total_params, "Trainable params": trainable_params}

    def forward(self, batch: list):
        device = next(self.parameters()).device
        data_list = []

        for item in batch:
            data_list.append(Data(x=item.pyg_data.x, edge_index=item.pyg_data.edge_index))

        merged = Batch.from_data_list(data_list).to(device)
        x, edge_index, batch_vec = merged.x, merged.edge_index, merged.batch

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

        logits = self.mlp(torch.cat([skip, x], dim=1)) # [BATCH, 288] -> [BATCH, 64] -> [BATCH, 2], the final prediction
        return logits

class TextData():
    def __init__(self, question, chunks, y):
        self.question = question
        self.chunks = chunks
        self.y = y

class TextDataset(Dataset):
    def __init__(self, tuples: list):
        self.tuples = tuples

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, pos):
        question, chunks, label = self.tuples[pos]
        y = torch.tensor([label], dtype=torch.long)
        return TextData(question, chunks, y)

class DirectEmbedder(nn.Module):
    def __init__(self, bert_model_name: str = None):
        super().__init__()
        self.tokenizer = graphFunctions.tokenizer
        self.bert = AutoModel.from_pretrained(graphFunctions.BERT_MODEL)

        self.max_seq_len = 512      # BERT's absolute limit
        self.question_len = 128
        self.window_len = self.max_seq_len - self.question_len - 1  # -1 for trailing [SEP]
        self.stride = 32

    def _split_into_windows(self, token_ids, attention_mask):
        """Split a long tokenized sequence into overlapping windows."""
        windows_ids, windows_mask = [], []
        start = 0
        while start < len(token_ids):
            end = min(start + self.window_len, len(token_ids))
            windows_ids.append(token_ids[start:end])
            windows_mask.append(attention_mask[start:end])
            if end == len(token_ids):
                break
            start += self.window_len - self.stride  # overlap by `stride` tokens
        return windows_ids, windows_mask

    def forward(self, question: str, chunks: list[str]):
        device = next(self.bert.parameters()).device
        MAX_LEN = self.bert.config.max_position_embeddings
        H = self.bert.config.hidden_size

        enc_query = self.tokenizer(question, return_tensors="pt", truncation=False)
        query_ids = enc_query["input_ids"][0][1:-1]
        full_text = " ".join(chunks)
        
        enc_chunk = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False
        )
        chunk_ids = enc_chunk["input_ids"][0][1:-1]        # strip CLS and final SEP
        chunk_offsets = enc_chunk["offset_mapping"][0][1:-1]

        chunk_budget = MAX_LEN - 3 - len(query_ids)

        sep_id = torch.tensor([self.tokenizer.sep_token_id])
        cls_id = torch.tensor([self.tokenizer.cls_token_id])

        all_chunk_ids = chunk_ids
        num_chunk_tokens = len(all_chunk_ids)

        # step = max(1, chunk_budget // 2)
        step = max(1, chunk_budget * 3 // 4) # Passos mais largos
        window_start = 0

        summed = torch.zeros(H, device=device)
        total_count = 0

        while window_start < num_chunk_tokens:
            window_end = min(window_start + chunk_budget, num_chunk_tokens)
            window_ids = all_chunk_ids[window_start:window_end]

            input_ids = torch.cat([
                cls_id, query_ids, sep_id, window_ids, sep_id
            ]).unsqueeze(0).to(device)

            attention_mask = torch.ones_like(input_ids)
            type_0_len = 1 + len(query_ids) + 1
            type_1_len = len(window_ids) + 1
            token_type_ids = torch.cat([
                torch.zeros(type_0_len, dtype=torch.long),
                torch.ones(type_1_len, dtype=torch.long)
            ]).unsqueeze(0).to(device)

            hidden = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).last_hidden_state[0]

            summed += hidden.sum(dim=0)
            total_count += hidden.shape[0]

            if window_end == num_chunk_tokens:
                break
            window_start += step

        pooled = summed / max(total_count, 1) # Avoid division by zero, even though it shouldn't ever happen
        return pooled

class SemanticOnlyBERT(nn.Module):
    def __init__(
        self,
        bert_model_name:  str  = "bert-base-uncased",
        num_classes:      int  = 2,
        dropout_rate:     float = 0.1,
        freeze_bert_layers: int = 8,    # Freeze first N transformer layers
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.embedder = DirectEmbedder(bert_model_name)
        in_channels   = self.embedder.bert.config.hidden_size # 768 for default BERT

        self.freeze_bert_layers = freeze_bert_layers
        self._freeze_bert_layers(freeze_bert_layers) # Freeze BERT layers to avoid catastrophic forget and increase training performance

        # Classification head
        mlp_hidden    = in_channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
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

    def forward(self, batch):
        device = next(self.parameters()).device
        embeddings = []

        for item in batch:
            x = self.embedder(item.question, item.chunks)  # [H], one item at a time
            embeddings.append(x)

        x = torch.stack(embeddings).to(device)  # [BATCH, H]
        logits = self.mlp(x)                    # [BATCH, num_classes]
        return logits

class NodeDocData:
    def __init__(self, pyg_data, question: str, chunks: list[str], chunks_ids: list[torch.Tensor]):
        self.pyg_data   = pyg_data      # torch_geometric.data.Data  (x is spaCy placeholder)
        self.y = pyg_data.y
        self.question   = question
        self.chunks     = chunks
        self.chunks_ids = chunks_ids

# Container for easier accesss of the data 
class DocGraphDataset(Dataset):
    def __init__(self, tuples: list, cache=False): # Might turn the cache off if memory can't store it
        self.tuples = tuples
        self.cache = cache
        self._structure_cache = {} if cache else None

    def __len__(self):
        return len(self.tuples)

    def _build_structure(self, idx):
        question, chunks, label = self.tuples[idx]
        pyg_graph, chunks_ids = graphFunctions.process_doc_graph_instance(chunks)
        pyg_graph.y = torch.tensor([label], dtype=torch.long)
        return NodeDocData(pyg_graph, question=question, chunks=chunks, chunks_ids=chunks_ids)

    # TODO: REMOVE CACHE
    def __getitem__(self, pos):
        if self.cache: # If using cache
            if pos not in self._structure_cache: # Build the graph and save in cache in the first iteration, only returning the cached structure after that
                self._structure_cache[pos] = self._build_structure(pos)
            return self._structure_cache[pos]
        return self._build_structure(pos) # If not, always build the structure

# Process the text and generate embeddings to them, associating the embeddings with the graph nodes
class BERTDocEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.tokenizer = graphFunctions.tokenizer
        self.bert = AutoModel.from_pretrained(graphFunctions.BERT_MODEL) # bert_model_name
        # self.bert = graphFunctions.bert_model

    def forward(self, question: str, chunks_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.bert.parameters()).device
        MAX_LEN = self.bert.config.max_position_embeddings
        H = self.bert.config.hidden_size

        enc_query = self.tokenizer(question, return_tensors="pt", truncation=False)
        query_ids = enc_query["input_ids"][0][1:-1]

        chunk_budget = MAX_LEN - 3 - len(query_ids)

        sep_id = torch.tensor([self.tokenizer.sep_token_id])
        cls_id = torch.tensor([self.tokenizer.cls_token_id])


        # step = max(1, chunk_budget // 2)
        step = max(1, chunk_budget * 3 // 4) # Passos mais largos

        embeddings = []
        for chunk_ids in chunks_ids:
            valid_mask = chunk_ids != self.tokenizer.pad_token_id
            chunk_ids = chunk_ids[valid_mask]
            window_start = 0
            position_hidden: dict[int, list[torch.Tensor]] = {}

            num_chunk_tokens = len(chunk_ids)
            while window_start < num_chunk_tokens:
                window_end = min(window_start + chunk_budget, num_chunk_tokens)
                window_ids = chunk_ids[window_start:window_end]

                input_ids = torch.cat([
                    cls_id, query_ids, sep_id, window_ids, sep_id
                ]).unsqueeze(0).to(device)

                attention_mask = torch.ones_like(input_ids)
                type_0_len = 1 + len(query_ids) + 1
                type_1_len = len(window_ids) + 1
                token_type_ids = torch.cat([
                    torch.zeros(type_0_len, dtype=torch.long),
                    torch.ones(type_1_len, dtype=torch.long)
                ]).unsqueeze(0).to(device)

                hidden = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                ).last_hidden_state[0]

                for local_i, global_i in enumerate(range(window_start, window_end)):
                    position_hidden.setdefault(global_i, []).append(hidden[local_i + type_0_len])

                if window_end == num_chunk_tokens:
                    break
                window_start += step

            token_embeddings = torch.stack([
                torch.stack(vecs).mean(dim=0)
                for pos, vecs in sorted(position_hidden.items())
            ])
            chunk_embedding = token_embeddings.mean(dim=0)  # Shape: [H]
            embeddings.append(chunk_embedding)

        return torch.stack(embeddings)

class DocGATWithBERT(nn.Module):
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

        self.embedder = BERTDocEmbedder()
        in_channels   = self.embedder.bert.config.hidden_size # 768 for default BERT

        self.freeze_bert_layers = freeze_bert_layers
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
        data_list = []

        for item in batch:
            x = self.embedder(item.question, item.chunks_ids)
            data_list.append(Data(x=x, edge_index=item.pyg_data.edge_index))

        merged = Batch.from_data_list(data_list).to(device)
        x, edge_index, batch_vec = merged.x, merged.edge_index, merged.batch

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

        logits = self.mlp(torch.cat([skip, x], dim=1)) # [BATCH, 288] -> [BATCH, 64] -> [BATCH, 2], the final prediction
        return logits