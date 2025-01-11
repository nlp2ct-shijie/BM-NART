import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerModel
from .nat_ctc import NAT_ctc_model, NAT_ctc_decoder
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules import LayerNorm, MultiheadAttention, PositionalEmbedding
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from .TransformerBackwardDecoderLayer import TransformerBackwardDecoderLayer
from .TransformerBackwardDecoder import TransformerBackwardDecoder
import random
from contextlib import contextmanager
from torch_imputer import best_alignment
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


def convert_alignment_to_symbol(aligns, targets, blank, padding_mask, pad):
    def get_symbol(align, target):
        def _gs(a):
            if a % 2 == 0:
                symbol = blank
            else:
                symbol = target[a // 2]
            return symbol

        return list(map(_gs, align))

    symbols = torch.LongTensor(list(map(get_symbol, aligns, targets))).type_as(targets)
    symbols.masked_fill_(padding_mask, pad)
    return symbols


def get_at_dec_loss_mask(aligns, align_mask, targets, padding_mask):
    for i, align in enumerate(aligns):
        for j, a in enumerate(align):
            if a < 0 or a % 2 == 0:
                continue
            elif align_mask[i][j]:
                targets[i][a // 2] = True

    targets.masked_fill_(padding_mask, True)
    return targets

class ShallowTranformerForwardDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.no_encoder_attn = no_encoder_attn
        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.shallow_at_decoder_layers)
            ]
        )

class ShallowTranformerBackwardDecoder(TransformerBackwardDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.no_encoder_attn = no_encoder_attn
        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerBackwardDecoderLayer(args, no_encoder_attn)
                for _ in range(args.shallow_at_decoder_layers)
            ]
        )


class hybrid_decoder(NAT_ctc_decoder):
    def __init__(self, args, dictionary, embed_tokens,
                 for_at_dec_nat_enc,
                 for_at_dec_nat_dec_1,
                 for_at_dec_nat_dec_2,
                 for_at_dec_nat_dec_3,
                 for_at_dec_nat_dec_4,
                 back_at_dec_nat_enc,
                 back_at_dec_nat_dec_1,
                 back_at_dec_nat_dec_2,
                 back_at_dec_nat_dec_3,
                 back_at_dec_nat_dec_4,):
        super().__init__(args, dictionary, embed_tokens)
        self.for_at_dec_nat_enc = for_at_dec_nat_enc
        self.for_at_dec_nat_dec_1 = for_at_dec_nat_dec_1
        self.for_at_dec_nat_dec_2 = for_at_dec_nat_dec_2
        self.for_at_dec_nat_dec_3 = for_at_dec_nat_dec_3
        self.for_at_dec_nat_dec_4 = for_at_dec_nat_dec_4
        self.back_at_dec_nat_enc = back_at_dec_nat_enc
        self.back_at_dec_nat_dec_1 = back_at_dec_nat_dec_1
        self.back_at_dec_nat_dec_2 = back_at_dec_nat_dec_2
        self.back_at_dec_nat_dec_3 = back_at_dec_nat_dec_3
        self.back_at_dec_nat_dec_4 = back_at_dec_nat_dec_4
    
    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False):
        features, _ = self.extract_features(
            encoder_out=encoder_out,
            prev_output_tokens=prev_output_tokens
        )
        if features_only:  # used for mt_ctc_6_up_6
            return features, _
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


@register_model("BM_NART_GLAT_N4")
class BM_NART_GLAT_N4(NAT_ctc_model):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.dec_list = []
        self.for_at_dec_outputs = []
        self.back_at_dec_outputs = []
        self.encoder_out = None

        if args.at_direction == 'forward':
            if not (self.encoder.embed_tokens == self.decoder.embed_tokens
                 and self.encoder.embed_tokens == self.decoder.for_at_dec_nat_enc.embed_tokens):
                raise ValueError("The embedding layer isn't shared")
        elif args.at_direction == 'backward':
            if not (self.encoder.embed_tokens == self.decoder.embed_tokens
                 and self.encoder.embed_tokens == self.decoder.back_at_dec_nat_enc.embed_tokens):
                raise ValueError("The embedding layer isn't shared")
        elif args.at_direction == 'bidirection':
            if not (self.encoder.embed_tokens == self.decoder.embed_tokens
                 and self.encoder.embed_tokens == self.decoder.for_at_dec_nat_enc.embed_tokens
                 and self.decoder.for_at_dec_nat_enc.embed_tokens == self.decoder.back_at_dec_nat_enc.embed_tokens):
                raise ValueError("The embedding layer isn't shared")

        if args.share_all_embeddings and args.share_pos_embeddings:
            embed_dim = args.encoder_embed_dim
            padding_idx = args.pad
            embed_positions = (
                PositionalEmbedding(
                    args.max_source_positions,
                    embed_dim,
                    padding_idx,
                    learned=args.encoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )
            if args.at_direction == 'bidirection':
                for_dec_dict = {
                    0: self.decoder.for_at_dec_nat_dec_1,
                    1: self.decoder.for_at_dec_nat_dec_2,
                    2: self.decoder.for_at_dec_nat_dec_3,
                    3: self.decoder.for_at_dec_nat_dec_4
                }
                back_dec_dict = {
                    0: self.decoder.back_at_dec_nat_dec_1,
                    1: self.decoder.back_at_dec_nat_dec_2,
                    2: self.decoder.back_at_dec_nat_dec_3,
                    3: self.decoder.back_at_dec_nat_dec_4
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.for_at_dec_nat_enc.embed_positions = embed_positions
                self.decoder.back_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(4):
                    for_dec_dict[idx].embed_positions = embed_positions
                    back_dec_dict[idx].embed_positions = embed_positions
            elif args.at_direction == 'forward':
                for_dec_dict = {
                    0: self.decoder.for_at_dec_nat_dec_1,
                    1: self.decoder.for_at_dec_nat_dec_2,
                    2: self.decoder.for_at_dec_nat_dec_3,
                    3: self.decoder.for_at_dec_nat_dec_4
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.for_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(4):
                    for_dec_dict[idx].embed_positions = embed_positions
            elif args.at_direction == 'backward':
                back_dec_dict = {
                    0: self.decoder.back_at_dec_nat_dec_1,
                    1: self.decoder.back_at_dec_nat_dec_2,
                    2: self.decoder.back_at_dec_nat_dec_3,
                    3: self.decoder.back_at_dec_nat_dec_4
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.back_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(4):
                    back_dec_dict[idx].embed_positions = embed_positions
            else:
                raise NotImplementedError

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        q_noise = getattr(args, "quant_noise_pq", 0)
        quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        embed_dim = args.decoder_embed_dim
        export = getattr(args, "char_inputs", False)
        bias = True
        def share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, export, for_at_dec, back_at_dec):
            if getattr(args, "share_self_attn", False):
                for i in range(args.shallow_at_decoder_layers):
                    shared_q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, quant_noise_block_size)
                    shared_k_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, quant_noise_block_size)
                    shared_v_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, quant_noise_block_size)
                    shared_out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, quant_noise_block_size)

                    for_at_dec.layers[i].self_attn.q_proj = shared_q_proj
                    back_at_dec.layers[i].self_attn.q_proj = shared_q_proj
                    for_at_dec.layers[i].self_attn.k_proj = shared_k_proj
                    back_at_dec.layers[i].self_attn.k_proj = shared_k_proj
                    for_at_dec.layers[i].self_attn.v_proj = shared_v_proj
                    back_at_dec.layers[i].self_attn.v_proj = shared_v_proj
                    for_at_dec.layers[i].self_attn.out_proj = shared_out_proj
                    back_at_dec.layers[i].self_attn.out_proj = shared_out_proj

                    if getattr(args, "share_layernorm", False):
                        self_attn_layer_norm = LayerNorm(embed_dim, export=export)
                        for_at_dec.layers[i].self_attn_layer_norm = self_attn_layer_norm
                        back_at_dec.layers[i].self_attn_layer_norm = self_attn_layer_norm

            if getattr(args, "share_cross_attn", False):
                if not for_at_dec.no_encoder_attn and not back_at_dec.no_encoder_attn:
                    for i in range(args.shallow_at_decoder_layers):
                        shared_encoder_attn = cls.build_shared_encoder_attn(embed_dim, args)

                        for_at_dec.layers[i].encoder_attn = shared_encoder_attn
                        back_at_dec.layers[i].encoder_attn = shared_encoder_attn

                        if getattr(args, "share_layernorm", False):
                            encoder_attn_layer_norm = LayerNorm(embed_dim, export=export)
                            for_at_dec.layers[i].encoder_attn_layer_norm = encoder_attn_layer_norm
                            back_at_dec.layers[i].encoder_attn_layer_norm = encoder_attn_layer_norm

            if getattr(args, "share_ffn", False):
                for i in range(args.shallow_at_decoder_layers):
                    shared_fc1 = cls.build_shared_fc1(embed_dim, args.decoder_ffn_embed_dim, q_noise, quant_noise_block_size)
                    shared_fc2 = cls.build_shared_fc2(args.decoder_ffn_embed_dim, embed_dim, q_noise, quant_noise_block_size)

                    for_at_dec.layers[i].fc1 = shared_fc1
                    back_at_dec.layers[i].fc1 = shared_fc1
                    for_at_dec.layers[i].fc2 = shared_fc2
                    back_at_dec.layers[i].fc2 = shared_fc2

                    if getattr(args, "share_layernorm", False):
                        final_layer_norm = LayerNorm(embed_dim, export=export)
                        for_at_dec.layers[i].final_layer_norm = final_layer_norm
                        back_at_dec.layers[i].final_layer_norm = final_layer_norm

        if getattr(args, "share_at_decoder", False):
            if args.at_direction == 'forward':
                for_at_dec = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
                back_at_dec = None
            elif args.at_direction == 'backward':
                for_at_dec = None
                back_at_dec = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            elif args.at_direction == 'bidirection':
                for_at_dec = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
                back_at_dec = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
                share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, export, for_at_dec, back_at_dec)
            else:
                raise NotImplementedError

            return hybrid_decoder(args, tgt_dict, embed_tokens,
                                  for_at_dec, for_at_dec, for_at_dec, for_at_dec, for_at_dec,
                                  back_at_dec, back_at_dec, back_at_dec, back_at_dec, back_at_dec,)

        if args.at_direction == 'forward':
            for_at_dec_nat_enc = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_1 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_2 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_3 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_4 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_enc = None
            back_at_dec_nat_dec_1 = None
            back_at_dec_nat_dec_2 = None
            back_at_dec_nat_dec_3 = None
            back_at_dec_nat_dec_4 = None
        elif args.at_direction == 'backward':
            back_at_dec_nat_enc = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_1 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_2 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_3 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_4 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_enc = None
            for_at_dec_nat_dec_1 = None
            for_at_dec_nat_dec_2 = None
            for_at_dec_nat_dec_3 = None
            for_at_dec_nat_dec_4 = None
        elif args.at_direction == 'bidirection':
            for_at_dec_nat_enc = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_1 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_2 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_3 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_4 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_enc = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_1 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_2 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_3 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_4 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_enc, back_at_dec_nat_enc)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_1, back_at_dec_nat_dec_1)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_2, back_at_dec_nat_dec_2)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_3, back_at_dec_nat_dec_3)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_4, back_at_dec_nat_dec_4)
        else:
            raise NotImplementedError

        return hybrid_decoder(args, tgt_dict, embed_tokens,
                              for_at_dec_nat_enc,
                              for_at_dec_nat_dec_1,
                              for_at_dec_nat_dec_2,
                              for_at_dec_nat_dec_3,
                              for_at_dec_nat_dec_4,
                              back_at_dec_nat_enc,
                              back_at_dec_nat_dec_1,
                              back_at_dec_nat_dec_2,
                              back_at_dec_nat_dec_3,
                              back_at_dec_nat_dec_4
                              )

    def add_args(parser):
        NAT_ctc_model.add_args(parser)
        parser.add_argument("--shallow-at-decoder-layers", type=int, metavar='N',
                            help="the number of at decoder.")
        parser.add_argument("--share-at-decoder", default=False, action='store_true',
                            help='if set, share all at decoder\'s param.')
        parser.add_argument("--is-random", default=False, action='store_true',
                            help='if set, randomly select at decoder layer.')
        parser.add_argument("--glat-at-mask", default=False, action='store_true',
                            help='if set, compute at loss according to glat mask')
        parser.add_argument("--fast-glat", default=False, action='store_true',
                            help='if set, train glat faster')
        parser.add_argument("--without-enc", default=False, action='store_true',
                            help='if set, do not use nat encoder output.')
        parser.add_argument("--select-specific-at-decoder", nargs='+', type=int, default=[],
                            help='The specific at decoder chosen to use')
        parser.add_argument("--share-self-attn", default=False, action='store_true',
                            help='if set, share the self attention layer.')
        parser.add_argument("--share-cross-attn", default=False, action='store_true',
                            help='if set, share the encoder-decoder cross attention layer.')
        parser.add_argument("--share-ffn", default=False, action='store_true',
                            help='if set, share the FFN.')
        parser.add_argument("--share-layernorm", default=False, action='store_true',
                            help='if set, share the layernorm based on the previous three shared block.')
        parser.add_argument("--share-pos-embeddings", default=False, action='store_true',
                            help="if set, share the position embedding")
        parser.add_argument("--at-direction", default='bidirection', type=str,
                            help="the direction of at decoder.")

    def forward(self, at_src_tokens, nat_src_tokens, src_lengths, prev_nat, for_prev_at, back_prev_at, tgt_tokens, **kwargs):
        nat_encoder_out = self.encoder(nat_src_tokens, src_lengths=src_lengths, **kwargs)
        at_encoder_out = nat_encoder_out

        rand_seed = random.randint(0, 19260817)
        glat_info = None
        glat = kwargs['glat'] if 'glat' in kwargs else None
        glat_tgt_tokens = tgt_tokens
        if glat is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(
                        nat_encoder_out,
                        prev_nat,
                        normalize=False,
                    )
                pred_tokens = word_ins_out.argmax(-1)

                bsz = tgt_tokens.size(0)
                src_padding_mask = nat_encoder_out['encoder_padding_mask'][0]
                decoder_padding_mask = nat_encoder_out['upsample_mask']
                decoder_upsample_x = nat_encoder_out['upsample_x']
                seq_lens = (~decoder_padding_mask).sum(1)

                log_prob = F.log_softmax(word_ins_out, -1)
                nonpad_positions = tgt_tokens.ne(self.pad)
                tgt_lengths = nonpad_positions.sum(1)
                input_length = src_lengths * self.scale
                input_length = input_length.fill_(torch.max(input_length))
                best_aligns = best_alignment(log_prob.float(), tgt_tokens, input_length,
                                             tgt_lengths,
                                             self.blank_idx,
                                             True)
                aligned_tgt_tokens = convert_alignment_to_symbol(best_aligns, tgt_tokens, self.blank_idx,
                                                                 decoder_padding_mask, self.pad)

                same_target = pred_tokens.transpose(0, 1).eq(aligned_tgt_tokens)
                same_target = same_target.masked_fill(decoder_padding_mask, False)
                same_num = same_target.sum(-1)

                if not getattr(self.args, "fast_glat", False):
                    input_mask = torch.ones_like(prev_nat).fill_(1)
                    for li in range(bsz):
                        target_num = (((seq_lens[li] - same_num[li]).float()) * glat['context_p']).long()
                        if target_num > 0:
                            input_mask[li].scatter_(dim=0, index=torch.randperm(seq_lens[li])[:target_num].cuda(),
                                                    value=0)
                    input_mask = input_mask.eq(1)
                else:
                    keep_prob = ((seq_lens - same_num) / seq_lens * glat['context_p']).unsqueeze(-1)
                    input_mask = (
                            torch.rand(prev_nat.shape, device=word_ins_out.device) < keep_prob).bool()

                input_mask_1 = input_mask.masked_fill(decoder_padding_mask, False).unsqueeze(-1)
                a = decoder_upsample_x.masked_fill(~input_mask_1, 0)
                aligned_tgt_emb = self.decoder.embed_tokens(aligned_tgt_tokens)
                b = aligned_tgt_emb.masked_fill(input_mask_1, 0)
                glat_prev_output_emb = a + b

                if getattr(self.args, "glat_at_mask", False):
                    targets_mask = tgt_tokens.eq(self.pad).fill_(False)
                    targets_mask = get_at_dec_loss_mask(best_aligns, input_mask, targets_mask, ~nonpad_positions)
                    glat_tgt_tokens = tgt_tokens.masked_fill(targets_mask, self.pad)

                nat_encoder_out['upsample_x'] = glat_prev_output_emb
                at_encoder_out = nat_encoder_out

                glat_info = {
                                "glat_accu": (same_num.sum() / seq_lens.sum()).item(),
                                "glat_context_p": glat['context_p'],
                            }

        with torch_seed(rand_seed):
            nat_decode_features, dec_each_layer_output_and_attn = self.decoder(nat_encoder_out,
                                                                               prev_nat,
                                                                               normalize=False,
                                                                               features_only=True)

            nat_decode_output = self.decoder.output_layer(nat_decode_features)
            dec_each_layer_output = dec_each_layer_output_and_attn['inner_states']

            for_dec_dict = {
                1: self.decoder.for_at_dec_nat_dec_1,
                2: self.decoder.for_at_dec_nat_dec_2,
                3: self.decoder.for_at_dec_nat_dec_3,
                4: self.decoder.for_at_dec_nat_dec_4
            }
            back_dec_dict = {
                1: self.decoder.back_at_dec_nat_dec_1,
                2: self.decoder.back_at_dec_nat_dec_2,
                3: self.decoder.back_at_dec_nat_dec_3,
                4: self.decoder.back_at_dec_nat_dec_4
            }
            dec_list = [1, 2, 3, 4]
            if getattr(self.args, "is_random", False):
                random.shuffle(dec_list)
                dec_list = dec_list[:2]
                attr_list = getattr(self.args, "select_specific_at_decoder", [])
                if attr_list:
                    dec_list = attr_list
            self.dec_list = dec_list

            back_at_dec_output = []
            for_at_dec_output = []
            if self.args.at_direction == 'forward':
                if not getattr(self.args, "without_enc", False):
                    for_at_dec, _ = self.decoder.for_at_dec_nat_enc(for_prev_at,
                                                                    encoder_out=at_encoder_out,
                                                                    features_only=False,
                                                                    return_all_hiddens=False)
                    for_at_dec_output.append(for_at_dec)
                for idx, dec_layer_output in enumerate(dec_each_layer_output):
                    if idx not in dec_list:
                        continue
                    shallow_at_encode_output = {
                        "encoder_out": [dec_layer_output],
                        "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
                    }
                    for_at_dec, _ = for_dec_dict[idx](for_prev_at,
                                                      encoder_out=shallow_at_encode_output,
                                                      features_only=False,
                                                      return_all_hiddens=False)
                    for_at_dec_output.append(for_at_dec)
                self.for_at_dec_outputs = for_at_dec_output
            elif self.args.at_direction == 'backward':
                if not getattr(self.args, "without_enc", False):
                    back_at_dec, _ = self.decoder.back_at_dec_nat_enc(back_prev_at,
                                                                      encoder_out=at_encoder_out,
                                                                      features_only=False,
                                                                      return_all_hiddens=False)
                    back_at_dec_output.append(back_at_dec)
                for idx, dec_layer_output in enumerate(dec_each_layer_output):
                    if idx not in dec_list:
                        continue
                    shallow_at_encode_output = {
                        "encoder_out": [dec_layer_output],
                        "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
                    }
                    back_at_dec, _ = back_dec_dict[idx](back_prev_at,
                                                        encoder_out=shallow_at_encode_output,
                                                        features_only=False,
                                                        return_all_hiddens=False)
                    back_at_dec_output.append(back_at_dec)
                self.back_at_dec_outputs = back_at_dec_output
            elif self.args.at_direction == 'bidirection':
                if not getattr(self.args, "without_enc", False):
                    for_at_dec, _ = self.decoder.for_at_dec_nat_enc(for_prev_at,
                                                                    encoder_out=at_encoder_out,
                                                                    features_only=False,
                                                                    return_all_hiddens=False)
                    back_at_dec, _ = self.decoder.back_at_dec_nat_enc(back_prev_at,
                                                                      encoder_out=at_encoder_out,
                                                                      features_only=False,
                                                                      return_all_hiddens=False)
                    for_at_dec_output.append(for_at_dec)
                    back_at_dec_output.append(back_at_dec)
                for idx, dec_layer_output in enumerate(dec_each_layer_output):
                    if idx not in dec_list:
                        continue
                    shallow_at_encode_output = {
                        "encoder_out": [dec_layer_output],
                        "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
                    }
                    for_at_dec, _ = for_dec_dict[idx](for_prev_at,
                                                      encoder_out=shallow_at_encode_output,
                                                      features_only=False,
                                                      return_all_hiddens=False)
                    back_at_dec, _ = back_dec_dict[idx](back_prev_at,
                                                        encoder_out=shallow_at_encode_output,
                                                        features_only=False,
                                                        return_all_hiddens=False)
                    for_at_dec_output.append(for_at_dec)
                    back_at_dec_output.append(back_at_dec)
                self.for_at_dec_outputs = for_at_dec_output
                self.back_at_dec_outputs = back_at_dec_output

        for_at_loss, for_at_loss_list = self.compute_loss(for_at_dec_output, glat_tgt_tokens)
        back_at_loss, back_at_loss_list = self.compute_loss(back_at_dec_output, glat_tgt_tokens)
        ret = ({
                "out": nat_decode_output,  # T x B x C
                "name": "NAT"
            },
            {
                "loss": for_at_loss,  # B x T x C
                'at_loss_list': for_at_loss_list,
                "name": "FOR_AT"
            },
           {
               "loss": back_at_loss,  # B x T x C
               'at_loss_list': back_at_loss_list,
               "name": "BACK_AT"
           }
        )

        if glat is not None:
            ret[0].update(glat_info)
        return ret

    def compute_loss(self, at_net_outputs, at_target):
        at_loss_list = []
        masks = at_target.ne(self.pad)
        for i, at_net_output in enumerate(at_net_outputs):
            at_tgt = at_target
            if i > 0 and getattr(self.args, "glat_at_mask", False):
                at_net_output = at_net_output[masks]
                at_tgt = at_target[masks]
            at_lprobs = self.get_normalized_probs(at_net_output, log_probs=True)
            at_loss, _ = label_smoothed_nll_loss(
                at_lprobs.view(-1, at_lprobs.size(-1)), at_tgt.view(-1, 1), self.args.label_smoothing,
                ignore_index=self.pad,
                reduce=None,
            )
            at_loss = at_loss.mean()
            at_loss_list.append(at_loss)
        loss = sum(l for l in at_loss_list) / len(at_loss_list)
        return loss, at_loss_list

    @classmethod
    def build_shared_encoder_attn(cls, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=getattr(args, "quant_noise_pq", 0),
            qn_block_size=getattr(args, "quant_noise_pq_block_size", 8),
        )

    @classmethod
    def build_shared_fc1(cls, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    @classmethod
    def build_shared_fc2(cls, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)


@register_model_architecture("BM_NART_GLAT_N4", "BM_NART_GLAT_N4")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.upsample_scale = getattr(args, "upsample_scale", 3)
    args.shallow_at_decoder_layers = getattr(args, "shallow_at_decoder_layers", 1)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
