import sys
import os
import argparse
import re
import Inference_auto

def main(savedir, dataset, userdir, task, criterion, arch, max_token, max_updates, max_epochs, update_freq, output_bleu, saving_type,
         output_files, lr, warmup_updates, warmup_init_lr, at_weights, check_freq, with_curr, glat, save_interval_updates, no_epoch_checkpoint,
         stage_warmup_update, total_up):
    if saving_type == 'epoch':
        epoch_lst_len = len(max_epochs)
        epoch_lst_index = 0
        at_weight_index = 0
        at_weight = at_weights[at_weight_index]
        stage_update_times = get_stage_from_log(output_files)
        reset = False
        max_update = 0
        convert_para = at_weights[-2]
        while epoch_lst_index < epoch_lst_len:
            max_epoch = max_epochs[epoch_lst_index]
            train_order = get_train_order(savedir, dataset, userdir, task, criterion, arch, max_token, max_update, max_epoch, update_freq, output_bleu,
                              output_files, lr, warmup_updates, warmup_init_lr, at_weight, check_freq, with_curr, glat, save_interval_updates,
                              no_epoch_checkpoint, convert_para, total_up)
            os.system(train_order)
            is_change_stage, last_lr = monitor_log(output_files, stage_update_times)
            if is_change_stage:
                stage_update_times += 1
                at_weight_index += 1
                at_weight = at_weights[at_weight_index]
            else:
                epoch_lst_index += 1
                reset = False

                # tokenize_bleu_list
                resultfolder = 'results'
                if save_interval_updates != 0:
                    save_interval = True
                else:
                    save_interval = False
                Inference_auto.main(savedir, dataset, resultfolder, task, userdir, output_bleu, save_interval)

                if epoch_lst_index != epoch_lst_len:
                    # delete the useless checkpoint
                    delete_order = get_delete_order(savedir, output_bleu)
                    os.system(delete_order)

    elif saving_type == 'update':
        update_lst_len = len(max_updates)
        update_lst_index = 0
        at_weight_index = 0
        at_weight = at_weights[at_weight_index]
        stage_update_times = get_stage_from_log(output_files)
        reset = False
        max_epoch = 0
        convert_para = at_weights[-2]
        while update_lst_index < update_lst_len:
            max_update = max_updates[update_lst_index]
            train_order = get_train_order(savedir, dataset, userdir, task, criterion, arch, max_token, max_update,
                                          max_epoch, update_freq, output_bleu,
                                          output_files, lr, warmup_updates, warmup_init_lr, at_weight, check_freq,
                                          with_curr, glat, save_interval_updates,
                                          no_epoch_checkpoint, convert_para, total_up)
            os.system(train_order)
            is_change_stage, last_lr = monitor_log(output_files, stage_update_times)
            if is_change_stage:
                stage_update_times += 1
                at_weight_index += 1
                at_weight = at_weights[at_weight_index]
            else:
                update_lst_index += 1
                reset = False

                # tokenize_bleu_list
                resultfolder = 'results'
                if save_interval_updates != 0:
                    save_interval = True
                else:
                    save_interval = False
                Inference_auto.main(savedir, dataset, resultfolder, task, userdir, output_bleu, save_interval)

                if update_lst_index != update_lst_len:
                    # delete the useless checkpoint
                    delete_order = get_delete_order(savedir, output_bleu)
                    os.system(delete_order)
    else:
        raise ValueError('Please set the saving type correctly, choosing from \'epoch\' or \'update\'.')

    generate_order = get_generate_order(savedir, dataset, userdir, task, output_bleu)
    os.system(generate_order)

def get_train_order(savedir, dataset, userdir, task, criterion, arch, max_token, max_update, max_epoch, update_freq, output_bleu,
              output_files, lr, warmup_updates, warmup_init_lr, at_weight, check_freq, with_curr, glat, save_interval_updates,
              no_epoch_checkpoint,convert_para, total_up):
    str_files = ''
    for file in output_files:
        str_files += file + ' '
    order = 'python ../train.py ' + \
            dataset + \
            ' --save-dir ' + savedir + \
            ' --user-dir ' + userdir + \
            ' --arch ' + arch + \
            ' --task ' + task + \
            ' --criterion ' + criterion + \
            ' --fp16' + \
            ' --ddp-backend=no_c10d' + \
            ' --shallow-at-decoder-layers 1' + \
            ' --lambda-nat-at ' +  str(at_weight) + \
            ' --is-random' + \
            ' --share-at-decoder --share-self-attn --share-ffn --share-layernorm --share-pos-embeddings --select-specific-at-decoder 6' + \
            ' --noise full_mask' + \
            ' --share-all-embeddings' + \
            ' --optimizer adam --adam-betas \'(0.9,0.999)\'' + \
            ' --lr ' + str(lr) + ' --lr-scheduler inverse_sqrt ' + \
            ' --stop-min-lr \'1e-09\' --warmup-updates ' + str(warmup_updates) + \
            ' --warmup-init-lr \'' + str(warmup_init_lr) + '\' --label-smoothing 0.1' + \
            ' --dropout 0.3 --weight-decay 0.01' + \
            ' --decoder-learned-pos' + \
            ' --encoder-learned-pos' + \
            ' --apply-bert-init' + \
            ' --log-format \'simple\' --log-interval 100' + \
            ' --fixed-validation-seed 7' + \
            ' --max-tokens ' + str(max_token) + \
            ' --update-freq ' + str(update_freq) + \
            ' --max-epoch ' + str(max_epoch) + \
            ' --max-update ' + str(max_update) + \
            ' --total-up ' + str(total_up) + \
            ' --eval-bleu' + \
            ' --eval-bleu-args \'{\"beam\": 1, \"max_len_a\": 1.2, \"max_len_b\": 10}\'' + \
            ' --eval-bleu-detok moses' + \
            ' --eval-bleu-remove-bpe' + \
            ' --at-direction \'bidirection\' --log-paths ' + str_files + ' --check-freq ' + str(check_freq) + \
            ' --convert-para ' + str(convert_para)
    if with_curr:
        order += ' --with-curr '
    if glat:
        order += ' --glat '
    if save_interval_updates != 0:
        order += ' --save-interval-updates ' + str(save_interval_updates)
    if no_epoch_checkpoint:
        order += ' --no-epoch-checkpoints '
    return order

def get_delete_order(savedir, output_bleu):
    order = 'bleu_delete_point_names=$(sed -n \"21,\\$s/\\[\'//p\" ' + output_bleu + ' | sed \'s/\'\\\'\'.*//g\')\n' + \
            'echo bleu_delete_point_names \n' + \
            'echo $tok_bleu_delete_point_names \n' + \
            'for file in $tok_bleu_delete_point_names \n' + \
            'do \n' + \
            '    full_path=\"' + savedir + '/$file\" \n' + \
            '    if [ -f \"$full_path\" ]; then \n' + \
            '        rm \"$full_path\" \n' + \
            '        echo \"Deleted file: $full_path\" \n' + \
            '    else \n' + \
            '        echo \"File not found: $full_path\" \n' + \
            '    fi \n' + \
            'done\n'
    return order

def get_generate_order(savedir, dataset, userdir, task, output_bleu):
    order = 'bleu_avg_point_names=$(sed -n \'/\\[/,/\\]/p\' ' + output_bleu + ' | sed \'s/\\[\'\\\'\'//g;s/\'\\\'\'.*//g\' | head -n 10) \n' + \
            'echo $bleu_avg_point_names \n' + \
            'checkpoints=($bleu_avg_point_names) \n' + \
            'best5_inputs=\"\" \n' + \
            'for i in {0..4}; do \n' + \
            '    checkpoint=\"${checkpoints[$i]}\" \n' + \
            '    best5_inputs+=\"' + savedir + '/$checkpoint \" \n' + \
            'done \n\n' + \
            'python ../scripts/average_checkpoints.py --inputs $best5_inputs --output ' + savedir + '/checkpoint_best5.pt \n' + \
            'python ../generate.py' + \
            ' --path ' + savedir + '/checkpoint_best5.pt' + \
            ' ' + dataset + \
            ' --user-dir ' + userdir + \
            ' --gen-subset test' + \
            ' --task ' + task + \
            ' --iter-decode-max-iter 0' + \
            ' --iter-decode-eos-penalty 0' + \
            ' --beam 1' + \
            ' --remove-bpe' + \
            ' --print-step --max-tokens 1024 > gen.best5.out \n\n'
    return order

def monitor_log(log_files, stage_update_times):
    current_count = 0
    lr_value = 0
    for log_file in log_files:
        with open(log_file, 'r', encoding='UTF-8') as file:
            for line in file:
                if re.search(r'curriculum stage change.', line):
                    current_count += 1
                if re.search(r'(train) \| epoch', line):
                    lr_index = line.index('| lr')
                    start = lr_index + len('| lr ')
                    end = line.find(' ', start)
                    lr_value = float(line[start:end])
    print('stage_update_times:' + str(stage_update_times))
    print('current_count:' + str(current_count))
    if stage_update_times == current_count - 1:
        return True, lr_value
    elif stage_update_times == current_count:
        return False, lr_value
    else:
        raise ValueError('The count of stage update times appears to an error.')

def get_stage_from_log(log_files):
    stage_count = 0
    for log_file in log_files:
        with open(log_file, 'r', encoding='UTF-8') as file:
            for line in file:
                if re.search(r'curriculum stage change.', line):
                    stage_count += 1
    return stage_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, help='The path of the model folder. ', default=None)
    parser.add_argument('--dataset', type=str, help='The path of the dataset folder. ', default=None)
    parser.add_argument('--userdir', type=str, help='The path of the code folder. ', default=None)
    parser.add_argument('--task', type=str, help='The task name for training. ', default=None)
    parser.add_argument('--criterion', type=str, help='The criterion name for training. ', default=None)
    parser.add_argument('--arch', type=str, help='The arch name for training. ', default=None)
    parser.add_argument('--max-token', type=int, help='The max token for training. ', default=None)
    parser.add_argument('--max-updates', nargs='+', type=int, default=[], help='The max update list for training. ')
    parser.add_argument('--max-epochs', nargs='+', type=int, default=[], help='The max epoch list for training. ')
    parser.add_argument('--update-freq', type=int, help='The update freq for training. ', default=None)
    parser.add_argument('--output-bleu', type=str, help='The path of tokenizebleu log. ', default=None)
    parser.add_argument('--saving-type', type=str, help='Saving the checkpoint according to the epoch or the updates. ', default='epoch')
    parser.add_argument('--output-files', nargs='+', type=str, default=[], help='The path of the logfiles. The older it is, the earlier it should be put.')
    parser.add_argument('--lr', type=float, help='The learning rate for training. ', default=None)
    parser.add_argument('--warmup-updates', type=int, help='The warmup updates for training. ', default=None)
    parser.add_argument('--warmup-init-lr', type=float, help='The warmup initial learning rate for training. ', default=None)
    parser.add_argument('--at-weights', nargs='+', type=float, default=[], help='The total at weights will be used for training (sorted from the max one to the min one). ')
    parser.add_argument('--check-freq', type=int, help='The check frequency for curriculum learning. ', default=None)
    parser.add_argument('--save-interval-updates', type=int, help='The interval updates for saving a checkpoint. ', default=0)
    parser.add_argument('--with-curr', help='Whether apply the curriculum learning. ', default=False, action='store_true')
    parser.add_argument('--glat', help='Whether apply the glat. ', default=False, action='store_true')
    parser.add_argument('--no-epoch-checkpoint', help='Do not save the epoch checkpoint. ', default=False, action='store_true')
    parser.add_argument('--stage-warmup-update', type=int, help='The num of warmup update after the stage change. ', default=2000)
    parser.add_argument('--total-up', type=int, help='The num of total update of the experiment. ', default=300000)
    args = parser.parse_args()
    main(args.savedir, args.dataset, args.userdir, args.task, args.criterion, args.arch, args.max_token, args.max_updates, args.max_epochs, args.update_freq, \
         args.output_bleu, args.saving_type, args.output_files, args.lr, args.warmup_updates, args.warmup_init_lr, args.at_weights, args.check_freq, \
         args.with_curr, args.glat, args.save_interval_updates, args.no_epoch_checkpoint, args.stage_warmup_update, args.total_up)