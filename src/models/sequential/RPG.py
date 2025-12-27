# -*- coding: UTF-8 -*-
"""
RPG: Generating Long Semantic IDs in Parallel for Recommendation
Reference:
    "Generating Long Semantic IDs in Parallel for Recommendation"
    Yupeng Hou et al., KDD'2025.
"""

import os
import json
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 必须在导入transformers之前设置警告过滤器
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.utils._pytree.*')

from transformers import GPT2Config, GPT2Model
from tqdm import tqdm

from models.BaseModel import SequentialModel
from utils import utils


class ResBlock(nn.Module):
    """
    A Residual Block module.
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class RPGTokenizer:
    """
    RPG Tokenizer for generating semantic IDs
    """
    def __init__(self, args, corpus):
        self.args = args
        self.corpus = corpus
        self.n_codebook = args.n_codebook
        self.codebook_size = args.codebook_size
        self.n_codebook_bits = self._get_codebook_bits(self.codebook_size)
        self.item2tokens = self._init_tokenizer()
        self.eos_token = self.n_codebook * self.codebook_size + 1
        self.ignored_label = -100

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _init_tokenizer(self):
        """
        Initialize tokenizer by generating semantic IDs for items
        """
        # Check if semantic IDs already exist
        cache_dir = os.path.join(self.args.path, self.args.dataset, 'processed')
        os.makedirs(cache_dir, exist_ok=True)
        
        sem_ids_path = os.path.join(
            cache_dir,
            f'semantic_ids_n{self.n_codebook}_c{self.codebook_size}.json'
        )
        
        if os.path.exists(sem_ids_path):
            # Load existing semantic IDs
            with open(sem_ids_path, 'r') as f:
                item2sem_ids = json.load(f)
        else:
            # Generate semantic IDs using OPQ
            item2sem_ids = self._generate_semantic_ids()
            # Save semantic IDs
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)
        
        # Convert semantic IDs to tokens
        item2tokens = {}
        for item_id in range(1, self.corpus.n_items):
            if str(item_id) in item2sem_ids:
                tokens = list(item2sem_ids[str(item_id)])
            else:
                # Random tokens for items without semantic IDs
                tokens = [np.random.randint(0, self.codebook_size) for _ in range(self.n_codebook)]
            
            # Convert to token IDs: each digit gets offset
            token_ids = []
            for digit in range(self.n_codebook):
                token_ids.append(tokens[digit] + self.codebook_size * digit + 1)
            item2tokens[item_id] = tuple(token_ids)
        
        return item2tokens

    def _generate_semantic_ids(self):
        """
        Generate semantic IDs using OPQ (Product Quantization)
        This implementation uses item embeddings from a simple embedding layer
        """
        import logging
        logging.info("Generating semantic IDs using OPQ...")
        
        try:
            import faiss
            # Try to use faiss for OPQ
            # First, we need item embeddings - use a simple approach
            # In practice, you would use item metadata (text descriptions) to get embeddings
            
            # Generate random embeddings as placeholder
            # In real implementation, use sentence transformers or item metadata
            n_items = self.corpus.n_items - 1
            emb_dim = 128  # Embedding dimension for OPQ
            
            # Create random embeddings (in practice, use item metadata embeddings)
            item_embs = np.random.randn(n_items, emb_dim).astype('float32')
            item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
            
            # Build OPQ index
            n_codebook_bits = self._get_codebook_bits(self.codebook_size)
            index_factory = f'OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{n_codebook_bits}'
            
            index = faiss.index_factory(emb_dim, index_factory, faiss.METRIC_INNER_PRODUCT)
            index.train(item_embs)
            index.add(item_embs)
            
            # Extract PQ codes
            ivf_index = faiss.downcast_index(index.index)
            invlists = faiss.extract_index_ivf(ivf_index).invlists
            ls = invlists.list_size(0)
            pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
            pq_codes = pq_codes.reshape(-1, invlists.code_size)
            
            # Decode PQ codes to semantic IDs
            faiss_sem_ids = []
            n_bytes = pq_codes.shape[1]
            for u8code in pq_codes:
                bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
                code = []
                for i in range(self.n_codebook):
                    code.append(bs.read(n_codebook_bits))
                faiss_sem_ids.append(code)
            pq_codes_array = np.array(faiss_sem_ids)
            
            # Convert to dictionary
            item2sem_ids = {}
            for i in range(pq_codes_array.shape[0]):
                item_id = i + 1
                item2sem_ids[str(item_id)] = pq_codes_array[i].tolist()
            
            logging.info(f"Generated semantic IDs for {len(item2sem_ids)} items")
            return item2sem_ids
            
        except ImportError:
            import logging
            logging.warning("faiss not available, using random semantic IDs")
            # Fallback to random semantic IDs
            item2sem_ids = {}
            for item_id in range(1, self.corpus.n_items):
                tokens = [np.random.randint(0, self.codebook_size) for _ in range(self.n_codebook)]
                item2sem_ids[str(item_id)] = tokens
            return item2sem_ids

    @property
    def n_digit(self):
        return self.n_codebook

    @property
    def vocab_size(self):
        return self.eos_token + 1


class RPGBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--n_embd', type=int, default=448,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layer', type=int, default=2,
                            help='Number of transformer layers.')
        parser.add_argument('--n_head', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--n_inner', type=int, default=1024,
                            help='Inner dimension of feedforward network.')
        parser.add_argument('--activation_function', type=str, default='gelu_new',
                            help='Activation function.')
        parser.add_argument('--resid_pdrop', type=float, default=0.0,
                            help='Residual dropout probability.')
        parser.add_argument('--embd_pdrop', type=float, default=0.5,
                            help='Embedding dropout probability.')
        parser.add_argument('--attn_pdrop', type=float, default=0.5,
                            help='Attention dropout probability.')
        parser.add_argument('--layer_norm_epsilon', type=float, default=1e-12,
                            help='Layer norm epsilon.')
        parser.add_argument('--initializer_range', type=float, default=0.02,
                            help='Initializer range.')
        
        # Semantic ID configs
        parser.add_argument('--n_codebook', type=int, default=32,
                            help='Number of codebooks.')
        parser.add_argument('--codebook_size', type=int, default=256,
                            help='Size of each codebook.')
        
        # RPG configs
        parser.add_argument('--temperature', type=float, default=0.07,
                            help='Temperature for softmax.')
        parser.add_argument('--num_beams', type=int, default=50,
                            help='Number of beams for graph propagation.')
        parser.add_argument('--n_edges', type=int, default=50,
                            help='Number of edges in graph.')
        parser.add_argument('--propagation_steps', type=int, default=3,
                            help='Number of propagation steps.')
        parser.add_argument('--chunk_size', type=int, default=1024,
                            help='Chunk size for graph construction.')
        parser.add_argument('--use_graph_decoding', type=int, default=0,
                            help='Whether to use graph-constrained decoding.')
        
        return parser

    def _base_init(self, args, corpus):
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_inner = args.n_inner
        self.activation_function = args.activation_function
        self.resid_pdrop = args.resid_pdrop
        self.embd_pdrop = args.embd_pdrop
        self.attn_pdrop = args.attn_pdrop
        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.initializer_range = args.initializer_range
        
        self.n_codebook = args.n_codebook
        self.codebook_size = args.codebook_size
        self.temperature = args.temperature
        self.num_beams = args.num_beams
        self.n_edges = args.n_edges
        self.propagation_steps = args.propagation_steps
        self.chunk_size = args.chunk_size
        self.use_graph_decoding = args.use_graph_decoding
        
        # Initialize tokenizer
        self.tokenizer = RPGTokenizer(args, corpus)
        
        # Map item tokens
        self.item_id2tokens = self._map_item_tokens().to(self.device)
        
        # Initialize GPT2
        self._base_define_params()
        self.apply(self.init_weights)
        
        # Graph-constrained decoding
        self.generate_w_decoding_graph = False
        self.init_flag = False

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.
        """
        item_id2tokens = torch.zeros((self.item_num, self.tokenizer.n_digit), dtype=torch.long)
        for item_id in range(1, self.item_num):
            if item_id in self.tokenizer.item2tokens:
                item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item_id])
        return item_id2tokens

    def _base_define_params(self):
        gpt2config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            n_positions=self.history_max,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_inner=self.n_inner,
            activation_function=self.activation_function,
            resid_pdrop=self.resid_pdrop,
            embd_pdrop=self.embd_pdrop,
            attn_pdrop=self.attn_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            eos_token_id=self.tokenizer.eos_token,
        )
        
        self.gpt2 = GPT2Model(gpt2config)
        
        # Prediction heads
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.n_embd))
        self.pred_heads = nn.Sequential(*pred_head_list)
        
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.ignored_label)

    def forward(self, feed_dict):
        self.check_list = []
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape
        
        # Get input tokens for history items (handle padding)
        valid_history = history.clamp(min=0, max=self.item_num - 1)
        input_tokens = self.item_id2tokens[valid_history]  # [batch_size, seq_len, n_digit]
        
        # Average pooling over tokens to get item embeddings
        # Handle padding: only average over valid tokens
        input_embs_list = []
        for b in range(batch_size):
            valid_len = lengths[b].item()
            if valid_len > 0:
                tokens_b = input_tokens[b, :valid_len]  # [valid_len, n_digit]
                emb_b = self.gpt2.wte(tokens_b).mean(dim=-2)  # [valid_len, n_embd]
                # Pad to seq_len
                if valid_len < seq_len:
                    padding = torch.zeros(seq_len - valid_len, self.n_embd, device=self.device)
                    emb_b = torch.cat([emb_b, padding], dim=0)
                input_embs_list.append(emb_b)
            else:
                input_embs_list.append(torch.zeros(seq_len, self.n_embd, device=self.device))
        input_embs = torch.stack(input_embs_list, dim=0)  # [batch_size, seq_len, n_embd]
        
        # Create attention mask
        valid_mask = (history > 0).long()
        attention_mask = valid_mask
        
        # Forward through GPT2
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=attention_mask
        )
        
        # Get final states for each prediction head
        final_states = []
        for i in range(self.n_pred_head):
            state = self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
            final_states.append(state)
        final_states = torch.cat(final_states, dim=-2)  # [batch_size, seq_len, n_pred_head, n_embd]
        
        # Get prediction for the last position
        last_pos = (lengths - 1).clamp(min=0)
        last_states = final_states[torch.arange(batch_size), last_pos, :, :]  # [batch_size, n_pred_head, n_embd]
        last_states = F.normalize(last_states, dim=-1)
        
        # Compute logits for each codebook
        token_emb = self.gpt2.wte.weight[1:-1]  # [n_codebook * codebook_size, n_embd]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        
        # Get candidate items
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        i_ids_clamped = i_ids.clamp(min=0, max=self.item_num - 1)
        i_vectors = self.item_id2tokens[i_ids_clamped]  # [batch_size, n_candidates, n_digit]
        
        # Compute predictions
        predictions = []
        for i in range(self.n_pred_head):
            # Get token embeddings for this codebook
            token_emb_i = token_embs[i]  # [codebook_size, n_embd]
            # Compute logits
            logits_i = torch.matmul(last_states[:, i, :], token_emb_i.T) / self.temperature  # [batch_size, codebook_size]
            # Get token IDs for candidate items in this codebook
            token_ids_i = i_vectors[:, :, i] - i * self.codebook_size - 1  # [batch_size, n_candidates]
            token_ids_i = token_ids_i.clamp(min=0, max=self.codebook_size - 1)
            # Gather logits
            pred_i = torch.gather(logits_i.unsqueeze(1).expand(-1, i_ids.shape[1], -1), 
                                 dim=-1, index=token_ids_i.unsqueeze(-1)).squeeze(-1)
            predictions.append(pred_i)
        
        # Average predictions across all codebooks
        prediction = torch.stack(predictions, dim=0).mean(dim=0)  # [batch_size, n_candidates]
        
        out_dict = {'prediction': prediction}
        
        # For training: compute token prediction loss (as in original paper)
        # This is more aligned with the RPG paper's approach
        if self.training and feed_dict['phase'] == 'train':
            # Get target item (positive item, first candidate)
            target_items = i_ids[:, 0]  # [batch_size]
            target_tokens = self.item_id2tokens[target_items.clamp(0, self.item_num - 1)]  # [batch_size, n_digit]
            
            # Compute token logits for all codebooks (already computed above)
            # Use the logits computed for the last position
            token_logits_list = []
            for i in range(self.n_pred_head):
                token_emb_i = token_embs[i]  # [codebook_size, n_embd]
                logits_i = torch.matmul(last_states[:, i, :], token_emb_i.T) / self.temperature  # [batch_size, codebook_size]
                token_logits_list.append(logits_i)
            
            # Compute loss for each codebook
            losses = []
            for i in range(self.n_pred_head):
                # Convert target tokens to codebook indices
                target_token_indices = target_tokens[:, i] - i * self.codebook_size - 1
                target_token_indices = target_token_indices.clamp(0, self.codebook_size - 1)
                loss = self.loss_fct(token_logits_list[i], target_token_indices)
                losses.append(loss)
            
            # Average loss across all codebooks
            token_loss = torch.mean(torch.stack(losses))
            out_dict['token_loss'] = token_loss
        
        return out_dict

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        Compute loss for training.
        If token_loss is available (from forward), use it (aligned with original paper).
        Otherwise, fall back to BPR loss for compatibility.
        """
        if 'token_loss' in out_dict:
            # Use token prediction loss (as in original RPG paper)
            return out_dict['token_loss']
        
        # Fall back to BPR loss for backward compatibility
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        return loss

    def build_ii_sim_mat(self):
        """
        Build item-item similarity matrix for graph construction
        """
        n_items = self.item_num
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        
        # Reshape token embeddings
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)  # Convert to [0, 1]
        
        # Build similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.device, dtype=torch.float32)
        
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]
            
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]
                
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.device, dtype=torch.float32)
                
                for k in range(n_digit):
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1
                    row_inds = row_inds.clamp(0, codebook_size - 1)
                    col_inds = col_inds.clamp(0, codebook_size - 1)
                    
                    temp = token_sims_01[k].index_select(0, row_inds)
                    temp = temp.index_select(1, col_inds)
                    sum_block += temp
                
                avg_block = sum_block / n_digit
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block
        
        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        import logging
        logging.info("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        logging.info("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        """
        Graph-constrained decoding using propagation
        """
        batch_size = token_logits.shape[0]
        
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()
        
        # Randomly sample initial nodes
        topk_nodes_sorted = torch.randint(
            1, self.item_num,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )
        
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)
        
        for sid in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
            
            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)
                
                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)
                
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)
        
        return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1)


class RPG(SequentialModel, RPGBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['n_embd', 'n_layer', 'n_head', 'n_codebook', 'codebook_size', 'temperature']

    @staticmethod
    def parse_model_args(parser):
        parser = RPGBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = RPGBase.forward(self, feed_dict)
        result = {'prediction': out_dict['prediction']}
        if 'token_loss' in out_dict:
            result['token_loss'] = out_dict['token_loss']
        return result

    class Dataset(SequentialModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            # The base class already provides history_items, lengths, etc.
            return feed_dict
