"""
LLM-SRAF: LLM-Enhanced Semantic Resource Allocation Framework
大语言模型增强的卫星网络语义资源分配框架

Core modules:
- SemanticUnderstandingModule: 语义理解模块
- CrossModalFusion: 跨模态融合层
- ResourceDecisionModule: 资源决策模块
"""

from .model import (
    SemanticUnderstandingModule,
    CrossModalFusion,
    ResourceDecisionModule,
    LLMSRAF,
)

from .data import (
    SemanticPairDataset,
    SatelliteEnv,
    create_dataloader,
)

__version__ = "0.1.0"
__all__ = [
    "SemanticUnderstandingModule",
    "CrossModalFusion",
    "ResourceDecisionModule",
    "LLMSRAF",
    "SemanticPairDataset",
    "SatelliteEnv",
    "create_dataloader",
]