import torch
import torch.nn.functional as F
import math
import logging
import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
import numpy as np
import gc

logger = logging.getLogger(__name__)

@register_criterion("BM_NART_LOSS")
class BM_NART_LOSS(LabelSmoothedDualImitationCriterion):
    def __init__(self, task, lambda_nat_at, label_smoothing, zero_infinity, log_paths, check_freq, with_curr, convert_para, save_interval_updates):
        super().__init__(task, label_smoothing)
        self.lambda_nat_at = lambda_nat_at
        self.blank_idx = task.target_dictionary.blank_index
        self.upsample_scale = task.cfg.upsample_scale
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = zero_infinity

        self.interval_updates = save_interval_updates
        self.curriculum_stage = 1
        self.with_curr = with_curr
        self.convert_para = convert_para
        self.renew_flag = False
        self.check_freq = check_freq
        self.last_n_index = check_freq
        self.log_paths = log_paths
        if self.log_paths is None:
            raise ValueError("log_path should not be None")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--lambda-nat-at",
            default=0.5,
            type=float,
        )
        parser.add_argument(
            "--label-smoothing",
            default=0.1,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--with-curr",
            default=False,
            action='store_true',
            help="if set, enable the curriculum learning.",
        )
        parser.add_argument(
            "--convert-para",
            default=0.4,
            type=float,
            help="The at weight that need to convert to stage 3",
        )
        parser.add_argument(
            "--check-freq",
            default=3,
            type=int,
            help="The check_freq applied in the curriculum learning stage",
        )
        parser.add_argument(
            "--log-paths",
            nargs='+',
            type=str,
            default=[],
            help="the log file path",
        )
        parser.add_argument(
            '--zero-infinity', action='store_false', default=True)

    def forward(self, model, at_sample, nat_sample, total_up, reduce=True, **kwargs):
        logging_output={}
        nsentences, ntokens = at_sample["nsentences"], at_sample["ntokens"]
        # B x T
        at_src_tokens, src_lengths, nat_src_tokens = (
            at_sample["net_input"]["src_tokens"],
            at_sample["net_input"]["src_lengths"],
            nat_sample["net_input"]["src_tokens"]
        )
        at_tgt_tokens, prev_nat, for_prev_at, back_prev_at = at_sample["target"], \
                                                             nat_sample["prev_target"], \
                                                             at_sample["net_input"]["for_prev_output_tokens"], \
                                                             at_sample["net_input"]["back_prev_output_tokens"]

        hybrid_outputs = model(at_src_tokens, nat_src_tokens, src_lengths, prev_nat, for_prev_at, back_prev_at, at_tgt_tokens, **kwargs)
        hybrid_loss = {}

        for outputs in hybrid_outputs:
            if outputs['name'] == "NAT":
                net_output = outputs['out']
                lprobs = model.get_normalized_probs(
                    net_output, log_probs=True
                ).contiguous()
                input_lengths = nat_sample["net_input"]["src_lengths"] * self.upsample_scale
                nat_target = model.get_targets(nat_sample, net_output, "NAT")
                pad_mask = (nat_target != self.pad_idx) & (
                        nat_target != self.eos_idx
                )
                targets_flat = nat_target.masked_select(pad_mask)
                if "target_lengths" in nat_sample:
                    target_lengths = nat_sample["target_lengths"]
                else:
                    target_lengths = pad_mask.sum(-1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = F.ctc_loss(
                        lprobs.float(),  # to fix with fp16
                        targets_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="mean",
                        zero_infinity=self.zero_infinity,
                    )
                hybrid_loss["NAT"] = loss
            elif outputs['name'] == "FOR_AT":
                if model.args.at_direction == 'forward' or model.args.at_direction == 'bidirection':
                    if outputs.get("loss", None) is None:
                        at_net_outputs = outputs['out']
                        for_at_loss_list, for_at_nll_loss_list = [], []
                        output_property = outputs.get("property")
                        for i, at_net_output in enumerate(at_net_outputs):
                            at_lprobs = model.get_normalized_probs(at_net_output, log_probs=True)
                            if output_property is not None:
                                at_target = model.get_targets(at_sample, at_net_output, "AT", output_property[i])
                            else:
                                at_target = model.get_targets(at_sample, at_net_output, "AT")
                            at_loss, at_nll_loss = label_smoothed_nll_loss(
                                at_lprobs.view(-1, at_lprobs.size(-1)), at_target.view(-1, 1), self.label_smoothing,
                                ignore_index=self.padding_idx,
                                reduce=reduce,
                            )
                            at_loss, at_nll_loss = at_loss.mean(), at_nll_loss.mean()
                            for_at_loss_list.append(at_loss)
                            for_at_nll_loss_list.append(at_nll_loss)
                        hybrid_loss["FOR_AT"] = sum(l for l in for_at_loss_list) / len(for_at_loss_list)
                    else:
                        hybrid_loss["FOR_AT"] = outputs["loss"]
                        for_at_loss_list = outputs['at_loss_list']
            elif outputs['name'] == "BACK_AT":
                if model.args.at_direction == 'backward' or model.args.at_direction == 'bidirection':
                    if outputs.get("loss", None) is None:
                        at_net_outputs = outputs['out']
                        back_at_loss_list, back_at_nll_loss_list = [], []
                        output_property = outputs.get("property")
                        for i, at_net_output in enumerate(at_net_outputs):
                            at_lprobs = model.get_normalized_probs(at_net_output, log_probs=True)
                            if output_property is not None:
                                at_target = model.get_targets(at_sample, at_net_output, "BACK_AT", output_property[i])
                            else:
                                at_target = model.get_targets(at_sample, at_net_output, "BACK_AT")
                            at_loss, at_nll_loss = label_smoothed_nll_loss(
                                at_lprobs.view(-1, at_lprobs.size(-1)), at_target.view(-1, 1), self.label_smoothing,
                                ignore_index=self.padding_idx,
                                reduce=reduce,
                            )
                            at_loss, at_nll_loss = at_loss.mean(), at_nll_loss.mean()
                            back_at_loss_list.append(at_loss)
                            back_at_nll_loss_list.append(at_nll_loss)
                        hybrid_loss["BACK_AT"] = sum(l for l in back_at_loss_list) / len(back_at_loss_list)
                    else:
                        hybrid_loss["BACK_AT"] = outputs["loss"]
                        back_at_loss_list = outputs['at_loss_list']
            else:
                raise NotImplementedError

        def curriculum_lambda(log_paths, check_freq=3):
            if model.training:  # training period, don't change the loss weight
                if self.renew_flag == False:
                    self.renew_flag = True
                return self.lambda_nat_at
            else:  # valid period
                if self.renew_flag == True:
                    self.renew_flag = False
                    nat_bleu_lst, for_at_loss_lst, back_at_loss_lst, last_n_index_lst, curriculum_stage_lst, cl_at_para_lst = self.get_info_from_log(log_paths, self.interval_updates)
                    self.last_n_index = max(self.last_n_index, check_freq) if not last_n_index_lst else last_n_index_lst[-1]
                    self.curriculum_stage = self.curriculum_stage if not curriculum_stage_lst else curriculum_stage_lst[-1]
                    if self.curriculum_stage == 1:
                        if len(for_at_loss_lst) > self.last_n_index:
                            for_loss_last_n = [for_at_loss_lst[-i] for i in range(1, check_freq+1)]
                            back_loss_last_n = [back_at_loss_lst[-i] for i in range(1, check_freq+1)]
                            if min(for_loss_last_n) >= for_at_loss_lst[-(check_freq + 1)] and min(back_loss_last_n) >= back_at_loss_lst[-(check_freq + 1)]:
                                self.curriculum_stage += 1
                                self.last_n_index += check_freq + 1 + self.wait_for_warmup
                                logger.info("curriculum stage change.")
                                logger.info(f"The curriculum learning related parameter: last-n-index {self.last_n_index} | curriculum-stage {self.curriculum_stage} | curriculum-para {self.lambda_nat_at}")
                                exit()
                            self.last_n_index += 1
                    elif self.curriculum_stage == 2:
                        if len(nat_bleu_lst) > self.last_n_index:
                            bleu_loss_last_n = [nat_bleu_lst[-i] for i in range(1, check_freq + 1)]
                            if max(bleu_loss_last_n) <= nat_bleu_lst[-(check_freq + 1)]:
                                self.last_n_index += check_freq + 1 + self.wait_for_warmup
                                logger.info("curriculum stage change.")
                                if self.lambda_nat_at == self.convert_para:
                                    self.curriculum_stage += 1
                                logger.info(f"The curriculum learning related parameter: last-n-index {self.last_n_index} | curriculum-stage {self.curriculum_stage} | curriculum-para {self.lambda_nat_at}")
                                exit()
                            self.last_n_index += 1
                    elif self.curriculum_stage == 3:
                        self.last_n_index += 1
                    else:
                        raise NotImplementedError
                    logger.info(f"The curriculum learning related parameter: last-n-index {self.last_n_index} | curriculum-stage {self.curriculum_stage} | curriculum-para {self.lambda_nat_at}")
                return self.lambda_nat_at

        curr_lambda_nat_at = curriculum_lambda(self.log_paths, self.check_freq) if self.with_curr else self.lambda_nat_at
        
        if model.args.at_direction == 'bidirection':
            at_total_loss = (hybrid_loss["FOR_AT"] + hybrid_loss["BACK_AT"]) / 2
        elif model.args.at_direction == 'forward':
            at_total_loss = hybrid_loss["FOR_AT"]
        elif model.args.at_direction == 'backward':
            at_total_loss = hybrid_loss["BACK_AT"]
        else:
            raise NotImplementedError

        loss = curr_lambda_nat_at * at_total_loss + (1 - curr_lambda_nat_at) * hybrid_loss["NAT"]

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nat-ctc-loss": hybrid_loss["NAT"].data,
            "at-loss": at_total_loss.data,
        }
        if model.args.at_direction == 'forward':
            logging_output["for-at-average-loss"] = hybrid_loss["FOR_AT"].data
            num_at_loss = 1
            for for_at_loss in for_at_loss_list:
                logging_output["for-at-" + str(num_at_loss) + "-loss"] = for_at_loss.data
                num_at_loss += 1
        elif model.args.at_direction == 'backward':
            logging_output["back-at-average-loss"] = hybrid_loss["BACK_AT"].data
            num_at_loss = 1
            for back_at_loss in back_at_loss_list:
                logging_output["back-at-" + str(num_at_loss) + "-loss"] = back_at_loss.data
                num_at_loss += 1
        elif model.args.at_direction == 'bidirection':
            logging_output["for-at-average-loss"] = hybrid_loss["FOR_AT"].data
            logging_output["back-at-average-loss"] = hybrid_loss["BACK_AT"].data
            num_at_loss = 1
            for for_at_loss in for_at_loss_list:
                logging_output["for-at-" + str(num_at_loss) + "-loss"] = for_at_loss.data
                num_at_loss += 1
            num_at_loss = 1
            for back_at_loss in back_at_loss_list:
                logging_output["back-at-" + str(num_at_loss) + "-loss"] = back_at_loss.data
                num_at_loss += 1
        if "glat_accu" in hybrid_outputs[0]:
            logging_output["glat_accu"] = hybrid_outputs[0]['glat_accu']
        if "glat_context_p" in hybrid_outputs[0]:
            logging_output['glat_context_p'] = hybrid_outputs[0]['glat_context_p']
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        def log_metric(key, logging_outputs):
            if len(logging_outputs) > 0 and key in logging_outputs[0]:
                metrics.log_scalar(
                    key,
                    utils.item(np.mean([torch.tensor(log.get(key, 0)).cpu() for log in logging_outputs])) / sample_size
                    if sample_size > 0 else 0.0,
                    sample_size,
                    round=3
                )

        log_metric("glat_accu", logging_outputs)
        log_metric("glat_context_p", logging_outputs)
        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    def get_info_from_log(self, log_paths, interval_updates):
        import re
        nat_bleu_lst = []
        for_loss_lst = []
        back_loss_lst = []
        last_n_index_lst = []
        cl_stage_lst = []
        cl_para_lst = []
        for log_path in log_paths:
            with open(log_path, 'r', encoding='UTF-8') as file:
                for line in file:
                    if re.search(r'The curriculum learning related parameter', line):
                        valid_line = file.readline()
                        
                        last_n_index = line.index('last-n-index')
                        start = last_n_index + len('last-n-index ')
                        end = line.find(' ', start)
                        last_n_index_value = int(line[start:end])
                        last_n_index_lst.append(last_n_index_value)

                        cl_stage_index = line.index('curriculum-stage')
                        start = cl_stage_index + len('curriculum-stage ')
                        end = line.find(' ', start)
                        cl_stage_value = int(line[start:end])
                        cl_stage_lst.append(cl_stage_value)

                        cl_para_index = line.index('curriculum-para')
                        start = cl_para_index + len('curriculum-para ')
                        end = line.find(' ', start)
                        cl_para_value = float(line[start:end])
                        cl_para_lst.append(cl_para_value)
                        
                        try:
                            updates_index = valid_line.index('| num_updates')
                            start = updates_index + len('| num_updates ')
                            end = valid_line.find(' ', start)
                            update_value = int(valid_line[start:end])
                        except:
                            continue
                        
                        if interval_updates != 0:
                            if update_value % interval_updates != 0:
                                continue
                        
                        bleu_index = valid_line.index('| bleu')
                        start = bleu_index + len('| bleu ')
                        end = valid_line.find(' ', start)
                        bleu_value = float(valid_line[start:end])
                        nat_bleu_lst.append(bleu_value)

                        for_loss_index = valid_line.index('for-at-average ')
                        start = for_loss_index + len('for-at-average ')
                        end = valid_line.find(' ', start)
                        for_loss_value = float(valid_line[start:end])
                        for_loss_lst.append(for_loss_value)

                        back_loss_index = valid_line.index('back-at-average ')
                        start = back_loss_index + len('back-at-average ')
                        end = valid_line.find(' ', start)
                        back_loss_value = float(valid_line[start:end])
                        back_loss_lst.append(back_loss_value)
                        
        return nat_bleu_lst, for_loss_lst, back_loss_lst, last_n_index_lst, cl_stage_lst, cl_para_lst
