"""HiCMAE encoders + DBM + one audio-neutral versus video-noise adversary.

Example additions to the usual data/checkpoint/output arguments:
  --condition-lambda 0.1 --dbm-visible-dim 128 --dbm-hidden-dims 256 128 --dbm-steps 5
"""
import math
import run_ravdess_interaction as experiment

LOGIT_TERMS = ('dbm',)
CONDITION_CLASSES = experiment.TEST_CONDITIONS  # both, video_noise, audio_neutral
PAIRWISE_CONDITIONS = {'audio_vs_av': 'audio_neutral', 'video_vs_av': 'video_noise'}
ADVERSARY_PAIRS = {'audio_vs_av': ('both', 'audio_neutral'),
                   'video_vs_av': ('both', 'video_noise'),
                   'audio_vs_video': ('audio_neutral', 'video_noise')}


def pairwise_targets(conditions, pairs=None):
    """0=first pair condition, 1=second, None=unrelated (excluded)."""
    if any(condition not in CONDITION_CLASSES for condition in conditions):
        raise ValueError('Unknown input condition')
    pairs = pairs if pairs is not None else {name: ADVERSARY_PAIRS[name] for name in PAIRWISE_CONDITIONS}
    return {name: [0 if condition == pair[0] else (1 if condition == pair[1] else None)
                   for condition in conditions] for name, pair in pairs.items()}


def pairwise_lambdas(args):
    result = {}
    for name, argument in (('audio_vs_av', 'audio_av_lambda'), ('video_vs_av', 'video_av_lambda')):
        value = getattr(args, argument, None)
        value = args.condition_lambda if value is None else value
        if not math.isfinite(value) or value < 0:
            raise ValueError('Pairwise adversarial weights must be finite and nonnegative')
        result[name] = value
    return result


def pairwise_cross_entropies(logits, conditions):
    """Compute each CE solely over its configured pair, including empty batches."""
    import torch
    import torch.nn.functional as F
    labels = pairwise_targets(conditions, {name: ADVERSARY_PAIRS[name] for name in logits})
    result = {}
    for name, scores in logits.items():
        targets = torch.tensor([-1 if value is None else value for value in labels[name]], device=scores.device)
        mask = targets >= 0
        values = F.cross_entropy(scores[mask], targets[mask], reduction='none') if mask.any() else scores.new_empty(0)
        result[name] = dict(targets=targets, mask=mask, losses=values)
    return result


def configure_parser(parser):
    parser.description = __doc__
    parser.add_argument('--condition-lambda', type=float, default=.1,
                        help='Reversed condition-loss gradient scale for encoders/DBM (default .1)')
    parser.add_argument('--condition-hidden-dim', type=int, default=128)
    parser.add_argument('--dbm-visible-dim', type=int, default=128)
    parser.add_argument('--dbm-hidden-dims', nargs=2, type=int, default=[256, 128], metavar=('H1', 'H2'))
    parser.add_argument('--dbm-steps', type=int, default=5,
                        help='Number of tied bidirectional mean-field refinement steps')
    parser.set_defaults(condition_classes=list(CONDITION_CLASSES), dbm_training='discriminative_mean_field',
                        condition_adversary_mode='audio_vs_video',
                        adversary_pairs={'audio_vs_video': list(ADVERSARY_PAIRS['audio_vs_video'])})


def validate_args(args):
    if not math.isfinite(args.condition_lambda) or args.condition_lambda < 0:
        raise ValueError('--condition-lambda must be finite and nonnegative')
    if getattr(args, 'condition_adversary_mode', None) == 'pairwise':
        coefficients = pairwise_lambdas(args)
        args.audio_av_lambda, args.video_av_lambda = coefficients['audio_vs_av'], coefficients['video_vs_av']
    if min(args.dbm_visible_dim, *args.dbm_hidden_dims, args.dbm_steps, args.condition_hidden_dim) < 1:
        raise ValueError('DBM dimensions, inference steps and condition hidden dimension must be positive')


def make_model(args):
    import torch
    from torch import nn
    from functools import partial
    from modeling_finetune_av import AudioVisionTransformer
    from dbm_fusion import MultimodalDBM, reverse_gradient

    class DBMEmotionModel(nn.Module):
        supports_condition_adversary = True

        def __init__(self):
            super().__init__()
            backbone = AudioVisionTransformer(
                img_size=160, patch_size=16, embed_dim=args.embed_dim,
                num_heads=args.embed_dim // 64, depth=args.depth,
                img_size_audio=(256, 128), patch_size_audio=16,
                embed_dim_audio=args.embed_dim, num_heads_audio=args.embed_dim // 64,
                depth_audio=args.depth, fusion_depth=0, num_classes=experiment.NUM_CLASSES,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                all_frames=16, drop_path_rate=args.drop_path)
            self.encoder, self.encoder_audio = backbone.encoder, backbone.encoder_audio
            self.video_layer_weights = backbone.video_layer_weights
            self.audio_layer_weights = backbone.audio_layer_weights
            # Encoder feature heads, not six-emotion logits, form DBM visibles.
            self.video_head = nn.Sequential(nn.Linear(args.embed_dim, args.dbm_visible_dim),
                                            nn.LayerNorm(args.dbm_visible_dim), nn.Tanh())
            self.audio_head = nn.Sequential(nn.Linear(args.embed_dim, args.dbm_visible_dim),
                                            nn.LayerNorm(args.dbm_visible_dim), nn.Tanh())
            self.dbm = MultimodalDBM(args.dbm_visible_dim, args.dbm_hidden_dims, args.dbm_steps)
            self.logit_terms = LOGIT_TERMS
            self.heads = nn.ModuleDict({'dbm': nn.Linear(args.dbm_hidden_dims[1], experiment.NUM_CLASSES, bias=False)})
            self.condition_lambda = args.condition_lambda
            mode = getattr(args, 'condition_adversary_mode', 'three_way')
            self.pairwise_adversaries = mode in ('pairwise', 'audio_vs_video')
            def classifier(classes):
                return nn.Sequential(nn.Linear(args.dbm_hidden_dims[1], args.condition_hidden_dim), nn.ReLU(),
                                     nn.Linear(args.condition_hidden_dim, classes))
            if self.pairwise_adversaries:
                self.condition_lambdas = ({'audio_vs_video': self.condition_lambda} if mode == 'audio_vs_video'
                                          else pairwise_lambdas(args))
                self.condition_classifiers = nn.ModuleDict({name: classifier(2) for name in self.condition_lambdas})
            else:
                self.condition_classifier = classifier(len(CONDITION_CLASSES))

        def forward(self, video, audio, return_terms=False, return_adversary=False):
            v = self.encoder(video).mean(1)
            a = self.encoder_audio(audio).mean(1)
            ov = (v * self.video_layer_weights.softmax(0)[None, :, None]).sum(1)
            oa = (a * self.audio_layer_weights.softmax(0)[None, :, None]).sum(1)
            fused = self.dbm(self.video_head(ov), self.audio_head(oa))
            logits = self.heads['dbm'](fused)
            terms = (logits,)
            if return_adversary:
                if self.pairwise_adversaries:
                    condition_logits = {name: head(reverse_gradient(fused, self.condition_lambdas[name]))
                                        for name, head in self.condition_classifiers.items()}
                else:
                    condition_logits = self.condition_classifier(reverse_gradient(fused, self.condition_lambda))
                return logits, terms, condition_logits
            return (logits, terms) if return_terms else logits

    return DBMEmotionModel()


def load_pretrained(model, path):
    """Reuse strict encoder loading; keep ALL DBM/adversary/projection heads new."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model', checkpoint.get('module', checkpoint))
    state = {(key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
    target = model.state_dict()
    prefixes = ('encoder.', 'encoder_audio.')
    required = [key for key in target if key.startswith(prefixes)]
    missing = [key for key in required if key not in state]
    mismatch = [key for key in required if key in state and state[key].shape != target[key].shape]
    if missing or mismatch:
        raise ValueError('Encoder checkpoint incompatible; missing: %s; shape mismatches: %s. Match --depth/--embed-dim.'
                         % (missing[:8], mismatch[:8]))
    kept = {key: value for key, value in state.items()
            if (key.startswith(prefixes) or key in ('video_layer_weights', 'audio_layer_weights'))
            and key in target and value.shape == target[key].shape}
    model.load_state_dict(kept, strict=False)
    print('Loaded %d encoder/pooling tensors; initialized DBM and condition/emotion classifiers.' % len(kept))


if __name__ == '__main__':
    experiment.main(model_factory=make_model, pretrained_loader=load_pretrained,
                    architecture='dbm', logit_terms=LOGIT_TERMS,
                    parser_configurator=configure_parser, args_validator=validate_args)
