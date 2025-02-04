import pandas as pd
import torchvision
from dataloader import *
from models.Bert import Bert
from models.Bertpt import Bertpt, SynAug ##AUG temporary in the same file of model
from models.NewBertpt import NewBertpt
from models.EVA import EVA_base, EVA_large
from models.Meta import Meta
from models.ModifiedResNet import ModifiedResNet
from models.ViT import ViT
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import albumentations as A


def merge_df_patches(df_patch:pd.DataFrame, df_meta:pd.DataFrame):
    return pd.merge(df_patch, df_meta, on=['Name', 'Names', 'Label'], how='right')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_scaler(scaler):
    if scaler == 'standard':
        return StandardScaler()
    else:
        return None
    
def balance_classes(df):
    max_class_count = df['Label'].value_counts().max()
    grouped = df.groupby('Label')
    resampled_df = grouped.apply(lambda x: x.sample(max_class_count, replace=True))
    resampled_df.reset_index(drop=True, inplace=True)
    return resampled_df
        

def get_loaders(PATH, df_:pd.DataFrame, test_df:pd.DataFrame, transform, scaler, batch_size, patch_mode):
    ## SCALER
    print(df_.shape)
    df_ = df_.dropna(axis=0)
    test_df = test_df.dropna(axis=0)
    print(df_.shape)
    
    scaler = get_scaler(scaler)
    df_.iloc[:, 3:] = scaler.fit_transform(df_.iloc[:, 3:])
    test_df.iloc[:, 3:] = scaler.transform(test_df.iloc[:, 3:])
    
    ## VAL SPLITING
    if patch_mode:
        df_["base"] = df_["Name"].apply(lambda x:x.split("__")[0])
        aux_df = df_.groupby("base", group_keys=True).apply(lambda x: x.iloc[0,:])
        train, val = train_test_split(aux_df, 
                                      random_state=42, 
                                      shuffle=True, 
                                      stratify=aux_df["Label"].values,
                                      test_size=0.2)
        train_df  = df_[df_["base"].isin(train["base"].values)]
        val_df = df_[df_["base"].isin(val["base"].values)]
        train_df.drop(['base'], axis=1, inplace=True)
        val_df.drop(['base'], axis=1, inplace=True)
    
    else:
        train_df, val_df = train_test_split(df_, 
                              random_state=42, 
                              shuffle=True, 
                              stratify=df_["Label"].values,
                              test_size=0.2)
    
    ## DATASETS
    trainds = LaminasAlbDS2(train_df,
                            PATH,
                            transform=transform,
                            img_format=patch_mode)
    valds = LaminasAlbDS2(val_df, PATH, img_format=patch_mode)
    testds = LaminasAlbDS2(test_df, PATH, img_format=patch_mode)
    
    ## DATALOADERS
    trainloader = DataLoader(trainds, 
                         batch_size,
                         collate_fn=collate_fn, 
                         num_workers=12)
    valloader = DataLoader(valds, 
                           batch_size, True,
                           collate_fn=collate_fn, 
                           num_workers=12)
    testloader = DataLoader(testds, batch_size, True,
                            collate_fn=collate_fn, 
                            num_workers=8)
    
    return trainloader, valloader, testloader


def get_textloaders(PATH, df_:pd.DataFrame, test_df:pd.DataFrame, norm_fn, transform, text_transform, batch_size, patch_mode, src_var=False, size=224, full_texts=False):


    train_df, val_df = train_test_split(df_, 
                          random_state=42, 
                          shuffle=True, 
                          stratify=df_["Label"].values,
                          test_size=0.2)
    
    ## DATASETS
    if src_var:
        trainds = LaminasTextDSVAR(train_df,
                                PATH,
                                norm_fn,
                                transform=transform,
                                img_format=patch_mode,
                                size=size)
        valds = LaminasTextDSVAR(val_df, PATH, norm_fn, img_format=patch_mode, size=size)
        testds = LaminasTextDSVAR(test_df, PATH, norm_fn, img_format=patch_mode, size=size)
     
    
    else:   
        if full_texts:
            print("Using full texts")
            trainds = LaminasTextDS_v2(train_df,
                                    PATH,
                                    norm_fn,
                                    transform=transform,
                                    text_transform = text_transform,
                                    img_format=patch_mode)
            valds = LaminasTextDS_v2(val_df, PATH, norm_fn, img_format=patch_mode)
            testds = LaminasTextDS_v2(test_df, PATH, norm_fn, img_format=patch_mode)
        else:

            trainds = LaminasTextDS(train_df,
                                    PATH,
                                    norm_fn,
                                    transform=transform,
                                    text_transform = text_transform,
                                    img_format=patch_mode)
            valds = LaminasTextDS(val_df, PATH, norm_fn, img_format=patch_mode)
            testds = LaminasTextDS(test_df, PATH, norm_fn, img_format=patch_mode)
    
    ## DATALOADERS
    trainloader = DataLoader(trainds, 
                         batch_size,
                         collate_fn=collate_fn, 
                         num_workers=0)
    valloader = DataLoader(valds, 
                           batch_size, True,
                           collate_fn=collate_fn, 
                           num_workers=0)
    testloader = DataLoader(testds, batch_size, True,
                            collate_fn=collate_fn, 
                            num_workers=0)
    
    return trainloader, valloader, testloader

def get_image_model(args):
    model = None
    if args.image_model == 'MODResnet':
        model = ModifiedResNet(layers=args.model_layers, 
                               output_dim=args.model_output, 
                               heads=args.heads, 
                               input_resolution=args.image_size,
                               width=args.width)
    

    elif args.image_model == 'ViT':
        model = ViT(output_dim=args.model_output,
                    feature_extracting=args.image_feature_extracting,
                    pretrained = args.vit_pretrained)
    elif args.image_model == 'EVA_base':
        model = EVA_base(output_dim=args.model_output,
                    feature_extracting=args.image_feature_extracting)
    elif args.image_model == 'EVA_large':
        model = EVA_large(output_dim=args.model_output,
                    feature_extracting=args.image_feature_extracting)
    return model 


def get_image_transform(args):
    transform = None
    if args.image_transform == 'geometric':
        transform = A.OneOf([
                            A.HorizontalFlip(p=1),
                            A.RandomRotate90(p=1),
                            A.VerticalFlip(p=1),
                            A.Transpose(p=1),
                            ], p=0.75)
    return transform

def get_text_transform(args):
    transform = None
    if args.text_transform == 'synonym':
        transform = SynAug(p = args.synonym_p, ratio = args.synonym_ratio)
    print(transform)
    return transform



def get_meta_model(args):
    model = None
    if args.meta_model == 'bert':
        model = Bert(output_dim = args.model_output)
    
    if args.meta_model == 'bertpt':
        model = Bertpt(output_dim = args.model_output)

    if args.meta_model == 'newbertpt':
        model = NewBertpt(output_dim = args.model_output)
        
    return model


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.lr_decay,
                                momentum=args.momentum,
                                nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.lr_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.lr_decay,
                                momentum=args.momentum,
                                alpha=args.alpha)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.lr_decay)
    else:
        raise ValueError("Unsupported optimizer")
    return optimizer
    

def get_scheduler(optimizer, args):
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    gamma=args.factor, 
                                                    step_size=args.patience)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0 = args.cosine_period, 
                                                    T_mult=1)
                                            
    if args.scheduler == 'exp':
        scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                    gamma=args.factor)

    return scheduler

def get_norm_fn(args):
    fn = None
    if args.image_model == 'MODResnet':
        fn = torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])
    

    elif args.image_model == 'ViT':
        fn = torchvision.transforms.Normalize([0.5, 0.5, 0.5], 
                                               [0.5, 0.5, 0.5])
        
    elif args.image_model == 'EVA_base':
        fn = torchvision.transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                                               [0.26862954, 0.26130258, 0.27577711])

    elif args.image_model == 'EVA_large':
        fn = torchvision.transforms.Normalize([0.48145466, 0.4578275, 0.40821073], 
                                               [0.26862954, 0.26130258, 0.27577711])
       
    return fn 