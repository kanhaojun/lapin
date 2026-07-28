import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs, save_imgs2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer,
                    wandb,
                    args):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
    epoch_losses = []
    loss_list = []
    for iter, data in enumerate(train_loader):
        t_start = time.time()
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        t_end = time.time()
        each_iter_time = t_end - t_start
        writer.add_scalar('loss', loss, global_step=step)
        if iter % config.print_interval == 0:
            if args.wandb:
                wandb.log({"train - epoch": epoch, "iter": iter, "loss": np.mean(loss_list), "lr": now_lr, "time": each_iter_time})  # Log loss, epoch, and step            
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}, time: {each_iter_time:.2f}s'
            print(log_info)
            logger.info(log_info)

    avg_loss = np.mean(loss_list)
    scheduler.step() 
    return step, avg_loss

def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    wandb,
                    args):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        if args.wandb:
            wandb.log({"interval - val epoch": epoch, "loss": np.mean(loss_list), "miou": miou, 
                       "f1_or_dsc" : f1_or_dsc, "accuracy": accuracy, "specificity": specificity, "sensitivity": sensitivity, "confusion_matrix": confusion}) 
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        if args.wandb:
            wandb.log({"val epoch": epoch, "loss": np.mean(loss_list)}) 
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)

def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    wandb,
                    args,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        if args.wandb:
            wandb.log({"test - best model - loss": np.mean(loss_list), "test - miou": miou, 
                       "test - f1_or_dsc" : f1_or_dsc, "test - accuracy": accuracy, "test - specificity": specificity, 
                       "test - sensitivity": sensitivity, "test - confusion_matrix": confusion}) 
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def train_one_epoch_fedavg(train_loader, model, criterion, optimizer, scheduler,
                           epoch, step, logger, config, writer, wandb, args):
    model.train()
    total_loss = 0
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            if args.wandb:
                wandb.log({
                    'train - epoch': epoch,
                    'iter': iter,
                    'loss': np.mean(loss_list),
                    'lr': now_lr,
                })
            log_info = (
                f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, '
                f'lr: {now_lr}'
            )
            print(log_info)
            logger.info(log_info)

    if scheduler is not None:
        scheduler.step()
    return model, total_loss / max(len(train_loader), 1)


def train_one_epoch_fedprox(train_loader, model, global_model, criterion, optimizer,
                            scheduler, epoch, step, logger, config, writer, mu,
                            wandb, args):
    model.train()
    total_loss = 0
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        proximal_term = 0.0
        for w, w_t in zip(model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        loss += (mu / 2) * proximal_term

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            if args.wandb:
                wandb.log({
                    'train - epoch': epoch,
                    'iter': iter,
                    'loss': np.mean(loss_list),
                    'lr': now_lr,
                })
            log_info = (
                f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, '
                f'lr: {now_lr}'
            )
            print(log_info)
            logger.info(log_info)

    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / max(len(train_loader), 1)
    if args.wandb:
        wandb.log({'avg_loss': avg_loss})
    return model, avg_loss


def train_one_epoch_scaffold(train_loader, model, criterion, optimizer, scheduler,
                             epoch, step, logger, config, writer, state_params_diff,
                             wandb, args):
    model.train()
    total_loss = 0
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)
        loss.backward()

        for param, diff in zip(model.parameters(), state_params_diff):
            if param.grad is not None:
                param.grad.data += diff.to(param.device)

        optimizer.step()
        total_loss += loss.item()
        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            avg_loss = np.mean(loss_list[-config.print_interval:])
            if args.wandb:
                wandb.log({
                    'train - epoch': epoch,
                    'iter': iter + 1,
                    'loss': avg_loss,
                    'lr': now_lr,
                })
            log_info = (
                f'train: epoch [{epoch}/{config.epochs}], '
                f'iter [{iter + 1}/{len(train_loader)}], loss: {avg_loss:.4f}, '
                f'lr: {now_lr:.6f}'
            )
            print(log_info)
            logger.info(log_info)

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(len(train_loader), 1)
    if args.wandb:
        wandb.log({'avg_loss': avg_loss})
    return model, avg_loss


def test_img(test_loader, model, criterion, epoch, logger, config, args):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze(1).cpu().detach().numpy())

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    log_info = (
        f'test_img epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
        f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, specificity: {specificity}, '
        f'sensitivity: {sensitivity}, confusion_matrix: {confusion}'
    )
    print(log_info)
    logger.info(log_info)
    return accuracy, miou, f1_or_dsc, np.mean(loss_list)
