import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerModel
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.modules import LayerNorm, MultiheadAttention, PositionalEmbedding
from .nat_ctc import NAT_ctc_model, NAT_ctc_decoder
from .TransformerBackwardDecoderLayer import TransformerBackwardDecoderLayer
from .TransformerBackwardDecoder import TransformerBackwardDecoder
import random


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
                 for_at_dec_nat_dec_5,
                 for_at_dec_nat_dec_6,
                 back_at_dec_nat_enc,
                 back_at_dec_nat_dec_1,
                 back_at_dec_nat_dec_2,
                 back_at_dec_nat_dec_3,
                 back_at_dec_nat_dec_4,
                 back_at_dec_nat_dec_5,
                 back_at_dec_nat_dec_6):
        super().__init__(args, dictionary, embed_tokens)
        self.for_at_dec_nat_enc = for_at_dec_nat_enc
        self.for_at_dec_nat_dec_1 = for_at_dec_nat_dec_1
        self.for_at_dec_nat_dec_2 = for_at_dec_nat_dec_2
        self.for_at_dec_nat_dec_3 = for_at_dec_nat_dec_3
        self.for_at_dec_nat_dec_4 = for_at_dec_nat_dec_4
        self.for_at_dec_nat_dec_5 = for_at_dec_nat_dec_5
        self.for_at_dec_nat_dec_6 = for_at_dec_nat_dec_6
        self.back_at_dec_nat_enc = back_at_dec_nat_enc
        self.back_at_dec_nat_dec_1 = back_at_dec_nat_dec_1
        self.back_at_dec_nat_dec_2 = back_at_dec_nat_dec_2
        self.back_at_dec_nat_dec_3 = back_at_dec_nat_dec_3
        self.back_at_dec_nat_dec_4 = back_at_dec_nat_dec_4
        self.back_at_dec_nat_dec_5 = back_at_dec_nat_dec_5
        self.back_at_dec_nat_dec_6 = back_at_dec_nat_dec_6

    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False):
        features, _ = self.extract_features(
            encoder_out=encoder_out,
            prev_output_tokens=prev_output_tokens
        )
        if features_only:
            return features, _
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


@register_model("BM_NART_CTC")
class BM_NART_CTC(NAT_ctc_model):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # Process the embedding layer
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
                    3: self.decoder.for_at_dec_nat_dec_4,
                    4: self.decoder.for_at_dec_nat_dec_5,
                    5: self.decoder.for_at_dec_nat_dec_6
                }
                back_dec_dict = {
                    0: self.decoder.back_at_dec_nat_dec_1,
                    1: self.decoder.back_at_dec_nat_dec_2,
                    2: self.decoder.back_at_dec_nat_dec_3,
                    3: self.decoder.back_at_dec_nat_dec_4,
                    4: self.decoder.back_at_dec_nat_dec_5,
                    5: self.decoder.back_at_dec_nat_dec_6
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.for_at_dec_nat_enc.embed_positions = embed_positions
                self.decoder.back_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(6):
                    for_dec_dict[idx].embed_positions = embed_positions
                    back_dec_dict[idx].embed_positions = embed_positions
            elif args.at_direction == 'forward':
                for_dec_dict = {
                    0: self.decoder.for_at_dec_nat_dec_1,
                    1: self.decoder.for_at_dec_nat_dec_2,
                    2: self.decoder.for_at_dec_nat_dec_3,
                    3: self.decoder.for_at_dec_nat_dec_4,
                    4: self.decoder.for_at_dec_nat_dec_5,
                    5: self.decoder.for_at_dec_nat_dec_6
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.for_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(6):
                    for_dec_dict[idx].embed_positions = embed_positions
            elif args.at_direction == 'backward':
                back_dec_dict = {
                    0: self.decoder.back_at_dec_nat_dec_1,
                    1: self.decoder.back_at_dec_nat_dec_2,
                    2: self.decoder.back_at_dec_nat_dec_3,
                    3: self.decoder.back_at_dec_nat_dec_4,
                    4: self.decoder.back_at_dec_nat_dec_5,
                    5: self.decoder.back_at_dec_nat_dec_6
                }
                self.encoder.embed_positions = embed_positions
                self.decoder.back_at_dec_nat_enc.embed_positions = embed_positions
                for idx in range(6):
                    back_dec_dict[idx].embed_positions = embed_positions
            else:
                raise NotImplementedError


    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        q_noise = getattr(args, "quant_noise_pq", 0)
        quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        embed_dim = args.decoder_embed_dim
        export = getattr(args, "char_inputs", False)
        bias=True

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
                    shared_fc1 = cls.build_shared_fc1(embed_dim, args.decoder_ffn_embed_dim, q_noise,quant_noise_block_size)
                    shared_fc2 = cls.build_shared_fc2(args.decoder_ffn_embed_dim, embed_dim, q_noise,quant_noise_block_size)

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
                                  for_at_dec, for_at_dec, for_at_dec, for_at_dec, for_at_dec, for_at_dec, for_at_dec,
                                  back_at_dec, back_at_dec, back_at_dec, back_at_dec, back_at_dec, back_at_dec, back_at_dec)

        if args.at_direction == 'forward':
            for_at_dec_nat_enc = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_1 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_2 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_3 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_4 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_5 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_6 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_enc = None
            back_at_dec_nat_dec_1 = None
            back_at_dec_nat_dec_2 = None
            back_at_dec_nat_dec_3 = None
            back_at_dec_nat_dec_4 = None
            back_at_dec_nat_dec_5 = None
            back_at_dec_nat_dec_6 = None
        elif args.at_direction == 'backward':
            back_at_dec_nat_enc = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_1 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_2 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_3 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_4 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_5 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_6 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_enc = None
            for_at_dec_nat_dec_1 = None
            for_at_dec_nat_dec_2 = None
            for_at_dec_nat_dec_3 = None
            for_at_dec_nat_dec_4 = None
            for_at_dec_nat_dec_5 = None
            for_at_dec_nat_dec_6 = None
        elif args.at_direction == 'bidirection':
            for_at_dec_nat_enc = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_1 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_2 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_3 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_4 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_5 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            for_at_dec_nat_dec_6 = ShallowTranformerForwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_enc = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_1 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_2 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_3 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_4 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_5 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            back_at_dec_nat_dec_6 = ShallowTranformerBackwardDecoder(args, tgt_dict, embed_tokens)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_enc, back_at_dec_nat_enc)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_1, back_at_dec_nat_dec_1)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_2, back_at_dec_nat_dec_2)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_3, back_at_dec_nat_dec_3)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_4, back_at_dec_nat_dec_4)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_5, back_at_dec_nat_dec_5)
            share_bid_layers(args, embed_dim, q_noise, quant_noise_block_size, for_at_dec_nat_dec_6, back_at_dec_nat_dec_6)
        else:
            raise NotImplementedError

        return hybrid_decoder(args, tgt_dict, embed_tokens,
                              for_at_dec_nat_enc,
                              for_at_dec_nat_dec_1,
                              for_at_dec_nat_dec_2,
                              for_at_dec_nat_dec_3,
                              for_at_dec_nat_dec_4,
                              for_at_dec_nat_dec_5,
                              for_at_dec_nat_dec_6,
                              back_at_dec_nat_enc,
                              back_at_dec_nat_dec_1,
                              back_at_dec_nat_dec_2,
                              back_at_dec_nat_dec_3,
                              back_at_dec_nat_dec_4,
                              back_at_dec_nat_dec_5,
                              back_at_dec_nat_dec_6
                              )

    def add_args(parser):
        NAT_ctc_model.add_args(parser)
        parser.add_argument("--shallow-at-decoder-layers", type=int, metavar='N',
                            help="the number of at decoder.")
        parser.add_argument("--share-at-decoder", default=False, action='store_true',
                            help='if set, share all at decoder\'s param.')
        parser.add_argument("--is-random", default=False, action='store_true',
                            help='if set, randomly select at decoder layer.')
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
        if getattr(self.args, "if_deepcopy_at_sample", False):
            at_encoder_out = self.encoder(at_src_tokens, src_lengths=src_lengths, **kwargs)
            nat_decode_output = self.decoder(nat_encoder_out,
                                             prev_nat,
                                             normalize=False,
                                             features_only=False)
            _, dec_each_layer_output_and_attn = self.decoder(at_encoder_out,
                                                             prev_nat,
                                                             normalize=False,
                                                             features_only=True)
        else:
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
            4: self.decoder.for_at_dec_nat_dec_4,
            5: self.decoder.for_at_dec_nat_dec_5,
            6: self.decoder.for_at_dec_nat_dec_6
        }
        back_dec_dict = {
            1: self.decoder.back_at_dec_nat_dec_1,
            2: self.decoder.back_at_dec_nat_dec_2,
            3: self.decoder.back_at_dec_nat_dec_3,
            4: self.decoder.back_at_dec_nat_dec_4,
            5: self.decoder.back_at_dec_nat_dec_5,
            6: self.decoder.back_at_dec_nat_dec_6
        }
        dec_list = [1, 2, 3, 4, 5, 6]
        if getattr(self.args, "is_random", False):
            random.shuffle(dec_list)
            dec_list = dec_list[:3]
            attr_list = getattr(self.args, "select_specific_at_decoder", [])
            if attr_list:
                dec_list = attr_list
        
        for_at_dec_output = []
        back_at_dec_output = []

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
        return ({
                    "out": nat_decode_output,  # T x B x C
                    "name": "NAT"
                },
                {
                    "out": for_at_dec_output,  # B x T x C
                    "name": "FOR_AT"
                },
                {
                    "out": back_at_dec_output,  # B x T x C
                    "name": "BACK_AT"
                },
        )


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

    @staticmethod
    def get_nat_decoder(model):
        return model.decoder

@register_model_architecture("BM_NART_CTC", "BM_NART_CTC")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
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
    args.decoder_layers = getattr(args, "decoder_layers", 6)
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
