# Modified from MLAVL (Xu et al., CVPR 2025), https://github.com/XuHuangbiao/MLAVL
# Changes: (1) ViFi-CLIP is only built when --use_clip is given; otherwise the precomputed text
#          feature file (--text_feature, torch.save format) is loaded once into a buffer.
#          (2) removed hard-coded .cuda() on w1/weight_AV/weight_V/edge_index so the model follows .to(device).
#          (3) Phase 3-3 regularisation (ours): --clip_mask p zeroes random whole clips (same mask for V and A) during
#              training, --input_dropout p is nn.Dropout on the projected V/A tokens before the shared encoder.
from torch import nn
import torch
import os
from models.transformer import Transformer
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import math
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=136):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = Variable(self.pe[:x.size(0), :], requires_grad=False)

        x = x + pe
        return self.dropout(x)


class MAGG_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(MAGG_Module, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, input_dim)
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight_AV)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


# model
class MLAVL(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, config, class_names):
        super(MLAVL, self).__init__()
        self.proj_visual = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        audio_dim = 768
        self.proj_audio = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=audio_dim, out_channels=audio_dim // 2),
            nn.BatchNorm1d(audio_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(kernel_size=1, in_channels=audio_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        text_dim = 512
        self.proj_text = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=text_dim, out_channels=text_dim // 2),
            nn.BatchNorm1d(text_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(kernel_size=1, in_channels=text_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.SSCEncoder = self.transformer.encoder
        self.GradeDecoder = self.transformer.decoder

        self.T_decoder = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=1,
            num_decoder_layers=2,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        ).decoder

        self.clip_fusion = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=hidden_dim * 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(kernel_size=1, in_channels=hidden_dim, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.use_pe = config.use_pe
        self.n_text = config.n_text
        self.n_query = config.n_query

        self.regressor_AV = nn.Linear(hidden_dim, self.n_query * 2)
        self.regressor_V = nn.Linear(hidden_dim, self.n_query)

        self.pos_encoder = PositionalEncoding(hidden_dim)

        # ours (Phase 3-3): input-side regularisation, both default to 0 = upstream behaviour
        self.clip_mask = float(getattr(config, 'clip_mask', 0.0) or 0.0)
        self.input_dropout = nn.Dropout(float(getattr(config, 'input_dropout', 0.0) or 0.0))

        self.w1 = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer('weight_AV', torch.linspace(0, 1, self.n_query * 2))
        self.register_buffer('weight_V', torch.linspace(0, 1, self.n_query))

        # Text features: either run ViFi-CLIP (upstream default) or load the precomputed file.
        if getattr(config, 'use_clip', False):
            from models.vificlip import returnCLIP
            self.CLIP = returnCLIP(config, class_names=class_names, )
            self.text_feature = None
        else:
            self.CLIP = None
            tf_path = getattr(config, 'text_feature', None)
            if tf_path is None:
                tf_path = {'fs1000': './FS_text_feature.npy', 'fisv': './FS_text_feature.npy',
                           'rg': './RG_text_feature.npy', 'logo': './LOGO_text_feature.npy'}[getattr(config, 'dataset', 'fs1000')]
            if not os.path.isfile(tf_path):
                raise FileNotFoundError(f'text feature not found: {tf_path} (pass --text_feature or --use_clip)')
            tf = torch.load(tf_path, map_location='cpu', weights_only=True).float()  # (2*n_text + 3*n_query, 512)
            expected = 2 * self.n_text + 3 * self.n_query
            if tf.shape[0] != expected:
                raise ValueError(f'{tf_path} has {tf.shape[0]} rows but --n_text {self.n_text} / --n_query {self.n_query} '
                                 f'expect {expected} (= 2*n_text + 3*n_query). Check --n_text (FS 42, RG 32).')
            self.register_buffer('text_feature', tf)
        self.gcn_V = MAGG_Module(hidden_dim, hidden_dim // 2, config.dropout)
        self.gcn_A = MAGG_Module(hidden_dim, hidden_dim // 2, config.dropout)

    def get_logits(self, inp, text):
        inp = inp / inp.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        logits = inp @ text.t()
        return logits

    def forward(self, visual, audio):
        # Shared-Specifc Context Encoder
        b, t, c = visual.shape
        if self.training and self.clip_mask > 0:  # ours: drop whole clips, aligned across modalities
            keep = (torch.rand(b, t, 1, device=visual.device) >= self.clip_mask).to(visual.dtype)
            visual = visual * keep
            audio = audio * keep
        visual = self.proj_visual(visual.transpose(1, 2)).transpose(1, 2)
        audio = self.proj_audio(audio.transpose(1, 2)).transpose(1, 2)
        visual = self.input_dropout(visual)  # ours: identity when --input_dropout 0
        audio = self.input_dropout(audio)

        encode_visual = self.SSCEncoder(visual)
        encode_audio = self.SSCEncoder(audio)
        encode_visual = encode_visual + visual
        encode_audio = encode_audio + audio

        # Get Text Feature
        text_feature = self.CLIP() if self.CLIP is not None else self.text_feature
        text_feature = text_feature.unsqueeze(0)
        text_feature = self.proj_text(text_feature.transpose(1, 2)).transpose(1, 2)
        text_feature = text_feature.repeat(b, 1, 1)
        text_fea_TV = text_feature[:, :self.n_text, :]
        text_fea_TA = text_feature[:, self.n_text:-3 * self.n_query, :]
        text_fea_AV = text_feature[:, -3 * self.n_query:-self.n_query, :]
        text_fea_V = text_feature[:, -self.n_query:, :]

        # Multidimensional Action Graph Guidance Module
        edges = []
        num_nodes = encode_visual.shape[1]
        num_nodes2 = text_fea_TV.shape[1]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        for i in range(num_nodes2):
            for j in range(num_nodes2):
                if i != j:
                    edges.append([num_nodes + i, num_nodes + j])
        for i in range(num_nodes):
            for j in range(num_nodes2):
                edges.append([num_nodes + j, i])
        edge_index = torch.tensor(edges, dtype=torch.long, device=visual.device).t().contiguous()

        fusion_TV = torch.cat([encode_visual, text_fea_TV], 1)
        mixed = []
        for batch_idx in range(fusion_TV.shape[0]):
            graph_visual = fusion_TV[batch_idx]
            mixed.append(self.gcn_V(graph_visual, edge_index))
        mixed = torch.stack(mixed)
        visual_fea, mixed1 = mixed[:, :num_nodes], mixed[:, num_nodes:]

        fusion_TA = torch.cat([encode_audio, text_fea_TA], 1)
        mixed = []
        for batch_idx in range(fusion_TA.shape[0]):
            graph_x = fusion_TA[batch_idx]
            mixed.append(self.gcn_A(graph_x, edge_index))
        mixed = torch.stack(mixed)
        audio_fea, mixed2 = mixed[:, :num_nodes], mixed[:, num_nodes:]

        # Audio-Visual Cross-Modal Fusion
        # Cross-Temporal Relation Decoder
        fusion_global, att_weights = self.T_decoder(visual_fea, audio_fea)
        # Clip-Wise Fusion Module
        fusion_clip = torch.cat([visual_fea, audio_fea], -1)
        fusion_clip = self.clip_fusion(fusion_clip.transpose(1, 2)).transpose(1, 2)
        fusion = fusion_clip + fusion_global

        if self.use_pe:
            prototype1 = self.pos_encoder(text_fea_AV)
            prototype2 = self.pos_encoder(text_fea_V)
        else:
            prototype1 = text_fea_AV
            prototype2 = text_fea_V
        # Performance Grading Transformer
        grade_AV, att_weights = self.GradeDecoder(prototype1, fusion)
        grade_V, att_weights = self.GradeDecoder(prototype2, visual_fea)

        # Get Logits for CE_Loss
        logits_AV = self.get_logits(grade_AV, text_fea_AV[0])
        logits_V = self.get_logits(grade_V, text_fea_V[0])

        # Score Generation
        s_AV = self.regressor_AV(grade_AV)  # (b, n, n)
        s_AV = torch.diagonal(s_AV, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s_AV)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out_AV = torch.sum(self.weight_AV.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        s_V = self.regressor_V(grade_V)  # (b, n, n)
        s_V = torch.diagonal(s_V, dim1=-2, dim2=-1)  # (b, n)
        norm_s2 = torch.sigmoid(s_V)
        norm_s2 = norm_s2 / torch.sum(norm_s2, dim=1, keepdim=True)
        out_V = torch.sum(self.weight_V.unsqueeze(0).repeat(b, 1) * norm_s2, dim=1)
        out = (self.w1 * out_AV) + ((1. - self.w1) * out_V)
        return {'output': out, 'embed': grade_AV, 'embed2': grade_V, 'embed3': text_fea_TV, 'embed4': text_fea_TA,
                'embed5': text_fea_AV, 'embed6': text_fea_V, 'logits1': logits_AV, 'logits2': logits_V}
