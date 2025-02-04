import os
import json
import time
import torch
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

from utils import *




def run_one_epoch(args, epoch, model, loaders, optimizer, criterion, device, scheduler=None,  best_loss=None, best_state_dict=None, pseudo_labels=None):

    print('EPOCH {}:'.format(epoch + 1))
    if args.freeze_epoch == epoch:
        model.meta_model.freeze()
        
    if args.unfreeze_epoch == epoch:
        model.meta_model.unfreeze()


    if epoch == 10:
        print('Freezing')
        model.meta_model.freeze(top_layers=4)
        model.image_model.freeze(top_layers=4)

    # train
    train_stats = train(epoch, model, loaders['train'], optimizer, criterion, device, pl = args.pseudo_labels, pseudo_labels=pseudo_labels, repeat_label=args.repeat_label)

    # logging
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

    if epoch == 0 or epoch % args.val_freq == 0 or epoch == args.epochs - 1:
        # validate
        val_stats, _, _, _= validate(model, loaders['val'], criterion, device, header='Val', repeat_label=args.repeat_label)

        # check for best
        if (best_loss is None) or (val_stats['loss'] < best_loss):
            print(f'saving new best checkpoint at epoch {epoch + 1}')
            best_loss = val_stats['loss']
            best_state_dict = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
            torch.save(best_state_dict, os.path.join(args.out_dir, "checkpoint-best.pth.tar"))

        log_stats = {**{k: v for k, v in log_stats.items()}, **{f'val_{k}': v for k, v in val_stats.items()}}

    with (Path(args.out_dir) / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

    if epoch == 0  and epoch % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
        print(f'saving checkpoint at epoch {epoch + 1}')
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(save_dict, os.path.join(args.out_dir, "checkpoint-last.pth.tar"))

    if scheduler is not None:
        scheduler.step()

    return log_stats, best_state_dict, best_loss


def train(epoch, model, loader, optimizer, criterion, device, log_every=1000, pl = False, pseudo_labels=None, repeat_label=False):
    torch.cuda.empty_cache()
    model.train()

    running_loss = 0.
    metrics = {}
    p_labels= None

    start_time = time.time()
    for batch_idx, (image, _, name, meta) in tqdm(enumerate(loader)):

        # move to gpu
        image = image.cuda(non_blocking=True)
        if isinstance(meta, torch.Tensor):
            meta = meta.cuda(non_blocking=True)
        
        # zero the gradients
        optimizer.zero_grad()

        # forward
        im_features, meta_features, log_scale = model(image, meta)

        if pl:
            p_labels = pseudo_labels['encoder_output'].last_hidden_state.detach().mean(dim=1) 
            p_labels = torch.nn.functional.cosine_similarity(p_labels.unsqueeze(1), p_labels.unsqueeze(0), dim=2)
            p_labels = p_labels**3

        # compute loss
        if repeat_label:
            loss = criterion(im_features, meta_features, log_scale, meta)
        else:
            loss = criterion(im_features, meta_features, log_scale, p_labels)

        # backward
        loss.backward()

        # step
        optimizer.step()
        
        # log 
        torch.cuda.synchronize()
        running_loss+=loss.item()
        
        if not((batch_idx+1)%log_every):
            print("[Train] batch {i} loss {loss:.3f} lr {lr:.8f}"
                  .format(epoch=epoch+1, i=(batch_idx+1), loss=running_loss/(batch_idx+1), lr=optimizer.param_groups[0]["lr"]))
    
    metrics["loss"] = running_loss/(batch_idx+1)

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print('* [Train] avg_loss {avg_loss:.3f} time {total_t}'
        .format(avg_loss=metrics['loss'], total_t=total_time_str))
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, header='Test', repeat_label=False):
    torch.cuda.empty_cache()
    model.eval()

    running_loss = 0.
    metrics = {}
    
    image_embs = []
    meta_embs = []
    names = []

    for batch_idx, (image, _, name, meta) in tqdm(enumerate(loader)):
        # move to gpu
        image = image.cuda(non_blocking=True)
        if isinstance(meta, torch.Tensor):
            meta = meta.cuda(non_blocking=True)
        

        # forward
        im_features, meta_features, log_scale = model(image, meta)

        # compute loss
        if repeat_label:
            loss = criterion(im_features, meta_features, log_scale, meta)
        else:
            loss = criterion(im_features, meta_features, log_scale)
        
        # append
        image_embs.append(im_features.detach().cpu())
        meta_embs.append(meta_features.detach().cpu())
        names.extend(name)

        # metrics
        torch.cuda.synchronize()
        running_loss+=loss.item()


    metrics["loss"] = running_loss/(batch_idx+1)
    


    print('* [avg_loss {avg_loss:.3f}'
        .format(avg_loss=metrics['loss']))


    return metrics, torch.concat(image_embs, dim=0).numpy(), torch.concat(meta_embs, dim=0).numpy(), names