import argparse
from utils import *
from train import *
from models.CLIP import CLIP
from loss import ClipLoss, ClipLoss_v2

import torch
import torchmetrics
import torchvision
from torchvision import transforms as T

import os
import json
import gc
import time


######## REMOVE ON DEBUG
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
########


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    
    ## LOAD OF DATA AND METADATA
    train_df = pd.read_csv(args.df.format('train'))
    test_df = pd.read_csv(args.df.format('test'))

    if args.gpt_aug:
        print("Integrating dataset augmented with chatgpt")
        train_df = pd.concat([train_df, pd.read_csv(r'chat_gpt_aug1.csv')]) #Augmentated description of private dataset
        train_df = pd.concat([train_df, pd.read_csv(r'chat_gpt_aug2.csv')]) #Augmentated description of private dataset, inserted two times to add variability
    
    if args.balance:
        train_df = balance_classes(train_df)

    print(train_df.columns)
    

    norm_fn = get_norm_fn(args)

    if args.image_size != 224:
        trainloader, valloader, testloader = get_textloaders(args.src_path,
                                                        train_df, 
                                                        test_df,
                                                        norm_fn = norm_fn,
                                                        transform = get_image_transform(args),
                                                        batch_size = args.batch_size,
                                                        patch_mode = 'patch' in args.df,
                                                        src_var = True,
                                                        size = args.image_size)
        
    else:
        trainloader, valloader, testloader = get_textloaders(args.src_path,
                                                        train_df, 
                                                        test_df,
                                                        norm_fn = norm_fn, 
                                                        transform = get_image_transform(args),
                                                        text_transform = get_text_transform(args),
                                                        batch_size = args.batch_size,
                                                        patch_mode = 'patch' in args.df,
                                                        full_texts = 'full' in args.df)

    
    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}
    
    criterion = ClipLoss(label_smoothing=args.label_smoothing)
    if args.repeat_label:
        criterion = ClipLoss_v2()
    
    if not(args.test):
        ## MODELS
        image_model = get_image_model(args)
        meta_model = get_meta_model(args)
        model = CLIP(image_model, meta_model)
        model = model.to(device)

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        

        best_loss = None
        best_state_dict = None
        start_time = time.time()
        pseudo_labels=None
        if args.pseudo_labels:
            pseudo_labels = {}
            def get_encoder_output(name):
                def hook(model, input, output):
                    pseudo_labels[name] = output
                return hook
            model.meta_model.bert.bert.encoder.register_forward_hook(get_encoder_output('encoder_output'))
        print(f"Start pseudo labels: {pseudo_labels}")

        for epoch in range(args.epochs):

            log_stats, best_state_dict, best_loss = run_one_epoch(args,
                epoch, model, dataloaders, optimizer, criterion, device, scheduler, best_loss, best_state_dict, pseudo_labels)

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('Training time: {}'.format(total_time_str))

        gc.collect()
    
    torch.cuda.synchronize()
    image_model = get_image_model(args)
    meta_model = get_meta_model(args)
    best_model = CLIP(image_model, meta_model)

    # load best checkpoint
    best_model_path = os.path.join(args.out_dir, "checkpoint-best.pth.tar")
    checkpoint = torch.load(best_model_path)
    try:
        msg = best_model.load_state_dict(checkpoint['state_dict'], strict=False)
    except:
        state_dict = best_state_dict.state_dict()
        msg = best_model.load_state_dict(state_dict, strict=False)
    print('pretrained weights found at {} and loaded with msg: {}'.format(best_model_path, msg))

    if torch.cuda.is_available():
        best_model.cuda()

    # run test
    test_stats, image_emb, meta_emb, names  = validate(best_model, dataloaders['test'], criterion, device, repeat_label=args.repeat_label)

    # logging
    with (Path(args.out_dir) / "test_log.txt").open("a") as f:
        f.write(json.dumps(test_stats) + "\n")

    # print stats
    for stat in test_stats:
        print(f"{stat}: {test_stats[stat]}")

    # save test embs
    df = pd.DataFrame.from_dict(data = np.concatenate((image_emb, meta_emb), axis=1))
    df['names'] = names
    df.to_csv(os.path.join(args.out_dir, "test_preds.csv"), index=None)
    
    
    # embs for train
    _, image_emb, meta_emb, names  = validate(best_model, dataloaders['train'], criterion, device, repeat_label=args.repeat_label)
    df_temp1 = pd.DataFrame.from_dict(data = np.concatenate((image_emb, meta_emb), axis=1))
    df_temp1['names'] = names
    _, image_emb, meta_emb, names  = validate(best_model, dataloaders['val'], criterion, device, repeat_label=args.repeat_label)
    df_temp2 = pd.DataFrame.from_dict(data = np.concatenate((image_emb, meta_emb), axis=1))
    df_temp2['names'] = names
    (pd.concat([df_temp1, df_temp2])).to_csv(os.path.join(args.out_dir, "train_preds.csv"), index=None)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Trainning")
    ## IO ARGS
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--save_ckpt_freq', type=int, default=10)
    parser.add_argument("--df", type=str, help="base name of dataframes(filenames and sentences)",
                        default=r"df_v6_sentences.csv",
                        required=False)
    parser.add_argument("--src_path", type=str, help="Path of the images",
                        default=r'dataset_v3',
                        required=False)
    parser.add_argument('--test', action='store_true', default=False, help='Test if experiment path was made already')    
    
    ## DATALOADER ARGS
    parser.add_argument('--balance', action='store_true', default=False, help='Balance classes for trainning')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_transform',
                        choices=['geometric', 'none'],
                        help='Augmentation for image',
                        default=None)
    
    parser.add_argument('--text_transform', choices=['synonym'], default=None, help='Augmentation for texts')
    parser.add_argument('--synonym_p', type=float, default=0.2, help='Rate of synonym replaces')
    parser.add_argument('--synonym_ratio', type=float, default=0.3, help='Rate of synonym replaces')
    parser.add_argument('--gpt_aug', action='store_true', default=False, help='Add gpt augmentations to train')



    ## MODEL ARGS
    parser.add_argument('--image_size', type=int, default=224, help='Embed dim for model image')
    parser.add_argument('--model_output', 
                        type=int, default=512,
                        help='Emb space for model output(both image and meta)')
    parser.add_argument('--image_model',
                        choices=['none', 'MODResnet', 'ViT', 'EVA_base', 'EVA_large'],
                        help='Model for the image',
                        default='MODResnet')
    parser.add_argument('--model_layers',
                        nargs='+',
                        default=[8, 8, 16, 32],
                        help='layers for modified resnet')
    parser.add_argument('--image_feature_extracting', action='store_true', default=False)
    parser.add_argument('--vit_pretrained',
                        choices=['imagenet', 'custom'],
                        help='Weights pretrained for ViT',
                        default=None)
    parser.add_argument('--heads', type=int, default=32, help='Heads for model image')
    parser.add_argument('--width', type=int, default=32, help='Embed dim for model image')
    parser.add_argument('--meta_model',
                        choices=['none', 'bert', 'bertpt', 'newbertpt'],
                        help='Model for the image',
                        default='bert')
    
    parser.add_argument('--freeze_epoch', type=int, default=None, help='Train only head of text image')
    parser.add_argument('--unfreeze_epoch', type=int, default=None, help='Train everything of text image')
    
    ## optimizer params
    parser.add_argument('--lr', type=float, default=1e-7, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-4, help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--alpha', type=float, default=0.8, help='smoothing parameter for rmsprop')
    parser.add_argument('--nesterov', action='store_true', default=False, help='Nesterov momentum')
    parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'rmsprop', 'adadelta'], default='sgd', help='Optimizers')
    parser.add_argument('--val_freq', type=int, default=1)
    
    ## scheduler params
    parser.add_argument('--scheduler', choices=['step', 'cosine', 'exp'], default=None, help='Schedulers')
    parser.add_argument('--factor', type=float, default=0.5, help='scheduler lr factor')
    parser.add_argument('--patience', type=int, default=5, help='scheduler step or patience')
    parser.add_argument('--cosine_period', type=int, default=10, help='Period for cosine scheduler')


    parser.add_argument('--label_smoothing', type=float, default=0.0)
    
    parser.add_argument('--pseudo_labels', action='store_true', default=False, help='Compute relation between sentences per batch to use for CELoss')

    parser.add_argument('--repeat_label', action='store_true', default=False, help='The labels are assigned different if there are repeated descriptions in a batch')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir + '/commandline_args.txt', 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)