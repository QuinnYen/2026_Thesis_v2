# BERT 編碼器
"""
BERT編碼器模組

提供基於BERT的文本編碼功能，支援：
- 預訓練BERT模型載入
- 文本序列編碼
- 方面感知編碼
- 層級特徵提取
- 注意力機制整合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path


class AspectAwareBERTEncoder(nn.Module):
    """
    方面感知的BERT編碼器
    
    整合方面資訊到BERT編碼過程中，提供更精確的方面級情感編碼
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 dropout_rate: float = 0.1,
                 freeze_bert: bool = False,
                 aspect_embedding_dim: int = 128):
        """
        初始化方面感知BERT編碼器
        
        Args:
            model_name: BERT模型名稱
            max_length: 最大序列長度
            dropout_rate: Dropout比率
            freeze_bert: 是否凍結BERT參數
            aspect_embedding_dim: 方面嵌入維度
        """
        super(AspectAwareBERTEncoder, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.freeze_bert = freeze_bert
        self.aspect_embedding_dim = aspect_embedding_dim
        
        # 載入BERT模型和分詞器
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 獲取BERT隱藏層維度
        self.hidden_size = self.bert_model.config.hidden_size
        
        # 方面嵌入層
        self.aspect_embedding = nn.Embedding(
            num_embeddings=100,  # 支援最多100個不同方面
            embedding_dim=aspect_embedding_dim
        )
        
        # 方面注意力層
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # 層正規化
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
        # 位置編碼增強器
        self.position_enhancer = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 方面融合層
        self.aspect_fusion = nn.Linear(
            self.hidden_size + aspect_embedding_dim, 
            self.hidden_size
        )
        
        # 是否凍結BERT參數
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                aspect_ids: Optional[torch.Tensor] = None,
                aspect_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            input_ids: 輸入token IDs [batch_size, seq_len]
            attention_mask: 注意力遮罩 [batch_size, seq_len]
            token_type_ids: token類型IDs [batch_size, seq_len]
            aspect_ids: 方面IDs [batch_size]
            aspect_positions: 方面位置 [batch_size, 2] (start, end)
        
        Returns:
            編碼結果字典
        """
        # BERT編碼
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 獲取序列輸出和池化輸出
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output      # [batch_size, hidden_size]
        
        # 位置編碼增強
        enhanced_sequence = self.position_enhancer(sequence_output)
        enhanced_sequence = self.layer_norm(enhanced_sequence + sequence_output)
        
        # 如果提供方面資訊，進行方面感知編碼
        if aspect_ids is not None and aspect_positions is not None:
            # 方面嵌入
            aspect_embeds = self.aspect_embedding(aspect_ids)  # [batch_size, aspect_dim]
            
            # 方面位置編碼
            aspect_aware_sequence = self._apply_aspect_attention(
                enhanced_sequence, aspect_embeds, aspect_positions, attention_mask
            )
            
            # 融合方面資訊
            batch_size, seq_len = enhanced_sequence.size()[:2]
            aspect_embeds_expanded = aspect_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            
            fused_features = torch.cat([aspect_aware_sequence, aspect_embeds_expanded], dim=-1)
            final_sequence = self.aspect_fusion(fused_features)
            final_sequence = self.dropout(final_sequence)
            
            # 更新池化輸出
            final_pooled = self._aspect_aware_pooling(final_sequence, aspect_positions, attention_mask)
        else:
            final_sequence = self.dropout(enhanced_sequence)
            final_pooled = pooled_output
        
        return {
            'sequence_output': final_sequence,      # [batch_size, seq_len, hidden_size]
            'pooled_output': final_pooled,         # [batch_size, hidden_size]
            'attention_weights': self._get_attention_weights(),  # 注意力權重
            'hidden_states': bert_outputs.hidden_states if hasattr(bert_outputs, 'hidden_states') else None
        }
    
    def _apply_aspect_attention(self, 
                               sequence_output: torch.Tensor,
                               aspect_embeds: torch.Tensor,
                               aspect_positions: torch.Tensor,
                               attention_mask: torch.Tensor) -> torch.Tensor:
        """
        應用方面注意力機制
        
        Args:
            sequence_output: 序列輸出 [batch_size, seq_len, hidden_size]
            aspect_embeds: 方面嵌入 [batch_size, aspect_dim]
            aspect_positions: 方面位置 [batch_size, 2]
            attention_mask: 注意力遮罩 [batch_size, seq_len]
        
        Returns:
            方面感知序列輸出
        """
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        # 建立方面查詢向量
        aspect_queries = self._create_aspect_queries(aspect_embeds, hidden_size)
        aspect_queries = aspect_queries.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 轉換為MultiheadAttention所需格式 [seq_len, batch_size, hidden_size]
        sequence_transposed = sequence_output.transpose(0, 1)
        aspect_queries_transposed = aspect_queries.transpose(0, 1)
        
        # 應用多頭注意力
        attended_output, attention_weights = self.aspect_attention(
            query=aspect_queries_transposed,
            key=sequence_transposed,
            value=sequence_transposed,
            key_padding_mask=~attention_mask.bool()  # 轉換遮罩格式
        )
        
        # 轉換回原格式
        attended_output = attended_output.transpose(0, 1)  # [batch_size, 1, hidden_size]
        
        # 廣播到整個序列
        attended_broadcast = attended_output.expand(-1, seq_len, -1)
        
        # 結合原始序列和注意力輸出
        aspect_aware_output = sequence_output + attended_broadcast
        aspect_aware_output = self.layer_norm(aspect_aware_output)
        
        return aspect_aware_output
    
    def _create_aspect_queries(self, aspect_embeds: torch.Tensor, hidden_size: int) -> torch.Tensor:
        """
        建立方面查詢向量
        
        Args:
            aspect_embeds: 方面嵌入 [batch_size, aspect_dim]
            hidden_size: 隱藏層大小
        
        Returns:
            方面查詢向量 [batch_size, hidden_size]
        """
        # 如果維度不匹配，進行投影
        if aspect_embeds.size(-1) != hidden_size:
            query_projection = nn.Linear(aspect_embeds.size(-1), hidden_size).to(aspect_embeds.device)
            aspect_queries = query_projection(aspect_embeds)
        else:
            aspect_queries = aspect_embeds
        
        return aspect_queries
    
    def _aspect_aware_pooling(self, 
                             sequence_output: torch.Tensor,
                             aspect_positions: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """
        方面感知池化
        
        Args:
            sequence_output: 序列輸出 [batch_size, seq_len, hidden_size]
            aspect_positions: 方面位置 [batch_size, 2]
            attention_mask: 注意力遮罩 [batch_size, seq_len]
        
        Returns:
            池化輸出 [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence_output.size()
        pooled_outputs = []
        
        for i in range(batch_size):
            start_pos, end_pos = aspect_positions[i]
            start_pos = max(0, min(start_pos.item(), seq_len - 1))
            end_pos = max(start_pos + 1, min(end_pos.item(), seq_len))
            
            # 提取方面相關的token
            aspect_tokens = sequence_output[i, start_pos:end_pos]  # [aspect_len, hidden_size]
            
            # 計算加權平均（考慮注意力遮罩）
            aspect_mask = attention_mask[i, start_pos:end_pos]
            if aspect_mask.sum() > 0:
                weights = aspect_mask.float() / aspect_mask.sum()
                aspect_pooled = torch.sum(aspect_tokens * weights.unsqueeze(-1), dim=0)
            else:
                aspect_pooled = torch.mean(aspect_tokens, dim=0)
            
            pooled_outputs.append(aspect_pooled)
        
        return torch.stack(pooled_outputs)
    
    def _get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        獲取注意力權重（如果可用）
        
        Returns:
            注意力權重或None
        """
        # 這裡可以返回最後一層的注意力權重
        # 實際實現可能需要修改BERT模型來保存注意力權重
        return None
    
    def encode_texts(self, 
                     texts: List[str],
                     aspects: Optional[List[str]] = None,
                     batch_size: int = 16) -> torch.Tensor:
        """
        批量編碼文本
        
        Args:
            texts: 文本列表
            aspects: 方面列表（可選）
            batch_size: 批次大小
        
        Returns:
            編碼結果 [num_texts, hidden_size]
        """
        self.eval()
        all_encodings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_aspects = aspects[i:i + batch_size] if aspects else None
                
                # 分詞和編碼
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 移動到設備
                input_ids = encoded['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
                
                # 前向傳播
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                all_encodings.append(outputs['pooled_output'])
        
        return torch.cat(all_encodings, dim=0)
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路徑
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型狀態
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'dropout_rate': self.dropout_rate,
                'freeze_bert': self.freeze_bert,
                'aspect_embedding_dim': self.aspect_embedding_dim
            }
        }, save_path / 'model.pth')
        
        # 保存分詞器
        self.tokenizer.save_pretrained(save_path / 'tokenizer')
    
    @classmethod
    def load_model(cls, load_path: str):
        """
        載入模型
        
        Args:
            load_path: 載入路徑
        
        Returns:
            載入的模型實例
        """
        load_path = Path(load_path)
        
        # 載入模型狀態
        checkpoint = torch.load(load_path / 'model.pth', map_location='cpu')
        config = checkpoint['config']
        
        # 建立模型實例
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 載入分詞器
        model.tokenizer = BertTokenizer.from_pretrained(load_path / 'tokenizer')
        
        return model


class HierarchicalBERTEncoder(nn.Module):
    """
    階層式BERT編碼器
    
    提供多層級的文本表示，包括詞級、句子級和文檔級編碼
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_layers: int = 3,
                 aggregation_method: str = 'attention'):
        """
        初始化階層式BERT編碼器
        
        Args:
            model_name: BERT模型名稱
            num_layers: 使用的BERT層數
            aggregation_method: 聚合方法 ('mean', 'max', 'attention')
        """
        super(HierarchicalBERTEncoder, self).__init__()
        
        self.model_name = model_name
        self.num_layers = num_layers
        self.aggregation_method = aggregation_method
        
        # 載入BERT模型
        self.bert_model = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True
        )
        
        self.hidden_size = self.bert_model.config.hidden_size
        
        # 層級聚合器
        if aggregation_method == 'attention':
            self.layer_attention = nn.Parameter(torch.ones(num_layers))
            self.layer_attention_activation = nn.Softmax(dim=0)
        
        # 句子級聚合器
        self.sentence_aggregator = nn.Linear(self.hidden_size, 1)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            input_ids: 輸入token IDs
            attention_mask: 注意力遮罩
            token_type_ids: token類型IDs
        
        Returns:
            階層式編碼結果
        """
        # BERT編碼
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        hidden_states = outputs.hidden_states  # Tuple of [batch_size, seq_len, hidden_size]
        attentions = outputs.attentions        # Tuple of [batch_size, num_heads, seq_len, seq_len]
        
        # 選擇指定數量的層
        selected_layers = hidden_states[-self.num_layers:]
        
        # 層級聚合
        if self.aggregation_method == 'mean':
            aggregated_hidden = torch.mean(torch.stack(selected_layers), dim=0)
        elif self.aggregation_method == 'max':
            aggregated_hidden = torch.max(torch.stack(selected_layers), dim=0)[0]
        elif self.aggregation_method == 'attention':
            layer_weights = self.layer_attention_activation(self.layer_attention)
            weighted_layers = torch.stack([
                layer_weights[i] * selected_layers[i] 
                for i in range(len(selected_layers))
            ])
            aggregated_hidden = torch.sum(weighted_layers, dim=0)
        
        # 句子級聚合
        sentence_weights = self.sentence_aggregator(aggregated_hidden)
        sentence_weights = F.softmax(sentence_weights, dim=1)
        sentence_representation = torch.sum(
            aggregated_hidden * sentence_weights, dim=1
        )
        
        return {
            'token_representations': aggregated_hidden,
            'sentence_representation': sentence_representation,
            'attention_weights': attentions,
            'layer_weights': layer_weights if self.aggregation_method == 'attention' else None
        }