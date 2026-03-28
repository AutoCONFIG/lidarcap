class GCSPFusion(nn.Module):
    def __init__(self, lidar_dim=1408, rgb_dim=512, hidden_dim=1024):
        raise NotImplementedError("多模态融合模块预留，待RGB数据验证后实现")
    
    def forward(self, lidar_feat, rgb_feat):
        raise NotImplementedError
