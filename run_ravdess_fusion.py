"""Original HiCMAE encoders with selectable visual/audio/fused output heads.

Uses the existing mixed-condition data, training, metric and explanation workflow.
Example:
  python run_ravdess_fusion.py --video-root /videos --audio-root /audio \
      --checkpoint /checkpoints/checkpoint-99.pth --output-dir runs/fusion01
"""
import run_ravdess_interaction as experiment

HEAD_NAMES = ('video', 'audio', 'fusion_video', 'fusion_audio')
LOGIT_TERMS = HEAD_NAMES


def normalize_heads(names):
    names = ['video' if name == 'visual' else name for name in names]
    if not names or any(name not in HEAD_NAMES for name in names):
        raise ValueError('Choose at least one head from visual, audio, fusion_video, fusion_audio')
    if len(set(names)) != len(names):
        raise ValueError('Each head may be selected only once (visual and video are aliases)')
    return tuple(name for name in HEAD_NAMES if name in names)


def make_model(args):
    import torch
    from torch import nn
    from functools import partial
    from modeling_finetune_av import AudioVisionTransformer

    class VideoFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            fusion_depth = getattr(args, 'fusion_depth', 2)
            if fusion_depth < 1:
                raise ValueError('The fusion experiment requires at least one fusion layer')
            backbone = AudioVisionTransformer(
                img_size=160, patch_size=16, embed_dim=args.embed_dim,
                num_heads=args.embed_dim // 64, depth=args.depth,
                img_size_audio=(256, 128), patch_size_audio=16,
                embed_dim_audio=args.embed_dim, num_heads_audio=args.embed_dim // 64,
                depth_audio=args.depth, fusion_depth=fusion_depth,
                fusion_num_heads=args.embed_dim // 64, num_classes=experiment.NUM_CLASSES,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                all_frames=16, drop_path_rate=args.drop_path)
            self.encoder = backbone.encoder
            self.encoder_audio = backbone.encoder_audio
            self.encoder_fusion = backbone.encoder_fusion
            self.legacy_combined_fusion = (not hasattr(args, 'heads') and
                                           tuple(getattr(args, 'logit_terms', ())) == ('video', 'fusion'))
            if self.legacy_combined_fusion:
                # Reconstruct checkpoints from the previous combined-fusion experiment.
                self.video_layer_weights = backbone.video_layer_weights
                self.logit_terms = ('video', 'fusion')
                self.wv = nn.Linear(args.embed_dim, experiment.NUM_CLASSES, bias=False)
                self.wf = nn.Linear(2 * args.embed_dim, experiment.NUM_CLASSES, bias=False)
            else:
                self.logit_terms = normalize_heads(getattr(args, 'heads', HEAD_NAMES))
                if 'video' in self.logit_terms:
                    self.video_layer_weights = backbone.video_layer_weights
                if 'audio' in self.logit_terms:
                    self.audio_layer_weights = backbone.audio_layer_weights
                self.heads = nn.ModuleDict({name: nn.Linear(args.embed_dim, experiment.NUM_CLASSES, bias=False)
                                            for name in self.logit_terms})

        def forward(self, video, audio, return_terms=False):
            needs_fusion = self.legacy_combined_fusion or any(name.startswith('fusion_') for name in self.logit_terms)
            features = {}
            if needs_fusion or 'video' in self.logit_terms:
                video_layers = self.encoder(video)  # B,Nv,L,D
            if needs_fusion or 'audio' in self.logit_terms:
                audio_layers = self.encoder_audio(audio)  # B,Na,L,D
            if 'video' in self.logit_terms:
                features['video'] = (video_layers.mean(1) * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            if 'audio' in self.logit_terms:
                features['audio'] = (audio_layers.mean(1) * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            if needs_fusion:
                fv, fa = self.encoder_fusion(video_layers[:, :, -1], audio_layers[:, :, -1])
                features['fusion_video'], features['fusion_audio'] = fv.mean(1), fa.mean(1)
            if self.legacy_combined_fusion:
                of = torch.cat((features['fusion_video'], features['fusion_audio']), dim=-1)
                terms = (self.wv(features['video']), self.wf(of))
            else:
                terms = tuple(self.heads[name](features[name]) for name in self.logit_terms)
            logits = sum(terms)
            return (logits, terms) if return_terms else logits

    return VideoFusionModel()


def load_pretrained(model, path):
    """Require matching pretrained weights for ALL THREE encoders."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('module', checkpoint))
    state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
    target = model.state_dict()
    prefixes = ('encoder.', 'encoder_audio.', 'encoder_fusion.')
    required = [key for key in target if key.startswith(prefixes)]
    missing = [key for key in required if key not in state]
    mismatched = [key for key in required if key in state and state[key].shape != target[key].shape]
    if missing or mismatched:
        raise ValueError('Fusion checkpoint incompatible; missing keys: %s; shape mismatches: %s. '
                         'Use the original HiCMAE pretrained checkpoint and match embedding/depth/fusion-depth.'
                         % (missing[:8], mismatched[:8]))
    kept = {key: value for key, value in state.items() if key in target
            and not key.startswith(('wv.', 'wa.', 'wf.', 'heads.')) and value.shape == target[key].shape}
    model.load_state_dict(kept, strict=False)
    print('Loaded %d encoder tensors; initialized selected heads: %s' % (len(kept), ', '.join(model.logit_terms)))


if __name__ == '__main__':
    experiment.main(model_factory=make_model, pretrained_loader=load_pretrained,
                    architecture='fusion', logit_terms=LOGIT_TERMS)
