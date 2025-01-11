import sys
import os
import argparse

def main(modelfolder, datafolder, resultfolder, task, user_dir, output_name, save_interval):
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    
    model_file_name_list = os.listdir(modelfolder)
    exist_file_name_list = os.listdir(resultfolder)
    
    model_file_suffix_list = [None for _ in model_file_name_list]
    exist_file_suffix_list = [None for _ in exist_file_name_list]
    for i in range(len(model_file_name_list)):
        model_file_suffix_list[i] = model_file_name_list[i].split('.')[0]
        model_file_suffix_list[i] = '_' + model_file_suffix_list[i].split('_', 1)[1] if '_' in model_file_suffix_list[i] else model_file_suffix_list[i].split('t')[1]
    for i in range(len(exist_file_name_list)):
        exist_file_suffix_list[i] = exist_file_name_list[i].split('.')[0]
        exist_file_suffix_list[i] = exist_file_suffix_list[i].split('_', 1)[1]
    
    if '_last' in model_file_suffix_list:
        model_file_suffix_list.remove('_last')
    if '_best' in model_file_suffix_list:
        model_file_suffix_list.remove('_best')
    if '_last' in exist_file_suffix_list:
        exist_file_suffix_list.remove('_last')
    if '_best' in exist_file_suffix_list:
        exist_file_suffix_list.remove('_best')
    
    generate_file_name_list = getGenerateList(model_file_suffix_list, exist_file_suffix_list)
    for suffix in generate_file_name_list:
        inferencecheckpoint(suffix,datafolder,modelfolder,task,user_dir,resultfolder)
    
    exist_file_name_list = os.listdir(resultfolder)
    exist_file_suffix_list = [None for _ in exist_file_name_list]
    for i in range(len(exist_file_name_list)):
        exist_file_suffix_list[i] = exist_file_name_list[i].split('.')[0]
        exist_file_suffix_list[i] = exist_file_suffix_list[i].split('_', 1)[1]
    BLEU_List = []
    for suffix in exist_file_suffix_list:
        checkpoint_name = 'checkpoint' + suffix + '.pt'
        #if checkpoint_name not in model_file_name_list:
            #continue
        BLEU = getcheckpointlist(suffix,resultfolder)
        BLEU_List.append([checkpoint_name,BLEU])
    
    best_checkpoint = AnalyzeResult(BLEU_List, save_interval)
    with open(output_name,'w') as f:
        for checkpoint in best_checkpoint:
            f.write(str(checkpoint) + '\n')
def AnalyzeResult(BLEU_List, save_interval):
    if save_interval:
        sorted_List = sorted(BLEU_List, reverse=1, key=lambda x: [x[1], int(x[0].split('checkpoint_')[1].split('_')[0]), int(x[0].split('_')[2].split('.')[0])])
    else:
        sorted_List = sorted(BLEU_List, reverse=1,key=lambda x: [x[1], int(x[0].split('checkpoint')[1].split('.')[0])])
    return sorted_List
def getkey(item):
    return item[1]
def getGenerateList(model_list, exist_list):
    generate_list = []
    for suffix in model_list:
        if suffix in exist_list:
            continue
        else:
            generate_list.append(suffix)
    if '_last' in generate_list:
        generate_list.remove('_last')
    if '_best' in generate_list:
        generate_list.remove('_best')
    return generate_list
def getcheckpointfilename(modelfolder):
    #get the file name list in the checkpointfile
    #raise error if the file doesn't end with .pt
    all_file = []
    for f in os.listdir(modelfolder):
        if (f[-1] != 't') | (f[-2] !='p'):
            print('error: checkpoint folder has file ends without .pt')
            exit()
        f_name = os.path.join(modelfolder,f)
        all_file.append(f_name)
    return all_file
def inferencecheckpoint(checkpoint_num,datafolder,modelfolder,task,user_dir,resultfolder):
    inference_order = 'python ../generate.py ' \
                        + datafolder+ \
                      ' --gen-subset valid ' \
                      '--task ' + task + \
                      ' --user-dir ' + user_dir + \
                      ' --path '+ modelfolder+'/checkpoint'+str(checkpoint_num)+'.pt' \
                      ' --remove-bpe --beam 1 --print-step --iter-decode-max-iter 0 --iter-decode-eos-penalty 0' \
                      ' --max-tokens 1024 > '+str(resultfolder)+'/valid_'+str(checkpoint_num)+'.out'
    os.system(inference_order)
def getcheckpointlist(checkpoint_num, resultfolder):
    with open(resultfolder+'/valid_'+str(checkpoint_num)+'.out') as f:
        lines = f.read().splitlines()
        lastline = lines[-1].replace(',', '').split()
        validbleu = float(lastline[6])
        return validbleu
    print('Error: result file not open')
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelfolder', help='The folder that saved the checkpoints', type=str)
    parser.add_argument('--datafolder', help='The folder that saved the dataset', type=str)
    parser.add_argument('--resultfolder', help='The folder that saved the inference results', type=str)
    parser.add_argument('--task', help='The task of the model', type=str)
    parser.add_argument('--user-dir', help='The folder of the model code', type=str)
    parser.add_argument('--output-name', help='The file that saves the log', type=str)
    parser.add_argument('--save-interval', help='whether save the interval', type=bool, default=False)
    args = parser.parse_args()
    main(args.modelfolder, args.datafolder, args.resultfolder, args.task, args.user_dir, args.output_name, args.save_interval)


