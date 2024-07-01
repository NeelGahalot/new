from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils import data, throughput_benchmark
from datasets import LeafDataset
from utils import ext_transforms as et
from metrics import StreamSegMetrics, BinarySegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.pyplot as plt

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_losss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):

        BCE = self.bce_losss(inputs, targets)

        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(
            inputs.sum() + targets.sum() + smooth)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class CombinedLoss(nn.Module):
    def __init__(self, temperature=1.0, kl_weight=1.0, dice_weight=20.0):
        super(CombinedLoss, self).__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.dice_weight = dice_weight
        self.dice_loss_fn = DiceBCELoss()

    def kl_divergence_loss(self, student_probs, teacher_probs):
        """
        Compute the KL Divergence loss with temperature scaling using probabilities.
        
        Args:
            student_probs: Probabilities from the student model (DeepLab).
            teacher_probs: Probabilities from the teacher model (SAM).
            
        Returns:
            Loss: KL Divergence loss.
        """
        # Apply temperature scaling
        student_probs = student_probs / self.temperature
        teacher_probs = teacher_probs / self.temperature
        
        # Compute log probabilities
        student_log_probs = torch.log(student_probs + 1e-8)
        
        # Compute KL Divergence loss
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        
        return loss

    def forward(self, student_logits, teacher_logits, inputs, targets):
        # Compute probabilities from logits
        #student_probs = F.softmax(student_logits, dim=1)
        #teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compute individual losses
        kl_loss = self.kl_divergence_loss(student_logits, teacher_logits)
        dice_loss = self.dice_loss_fn(inputs, targets)

        # Aggregate losses
        total_loss = self.kl_weight * kl_loss + self.dice_weight * dice_loss

        return total_loss


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='../',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='custom',
                        choices=['voc', 'cityscapes', 'custom'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--backbone_freeze_until", action='store_true', default=5)
    parser.add_argument("--generate_only", action='store_true', default= False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=10e2,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step', 'warmup', 'chained'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'BCE', 'binary_focal', 'binary_dice','kl_dice_combined'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # Custom Dataset Options
    parser.add_argument("--custom_data_path", type=str, default='../leaf_samples',
                        help="path to custom dataset")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--save_checkpoint", action='store_true', default=True,
                        help="Saving Checkpoint.")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'custom':
        """
        Augmentation to the custom dataset
        """
        train_img_transform = transforms.Compose([
          #RandomCropAndPad(512),
          transforms.Resize((512, 512)),
          #transforms.RandomResizedCrop(size=(256, 256)),
          #transforms.RandomHorizontalFlip(),
          #transforms.RandomRotation(degrees=(0, 360)),
          #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_mask_transform = transforms.Compose([
            #RandomCropAndPadMask(512),
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.RandomResizedCrop(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
        ])

        val_img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        train_dst = LeafDataset(root=opts.custom_data_path, image_set='train', img_transform=train_img_transform, mask_transform=train_mask_transform)
        val_dst = LeafDataset(root=opts.custom_data_path, image_set='val', img_transform=val_img_transform, mask_transform=val_mask_transform)


    return train_dst, val_dst


# ================================================= Validation ==============================================================

def generate(opts, model, loader, device):
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    img_id = 0
    
    with torch.no_grad():
        for i, (images) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            #labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            #preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            
            outputs = torch.squeeze(outputs, dim=1)
            #labels = labels.float()


            probs = torch.sigmoid(outputs).detach()
            preds = (probs > 0.5).long().cpu().numpy()
            
            #targets = labels.cpu().numpy()

            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                pred = preds[i]
                #target = targets[i]
                    # Decode the binary target and prediction masks to RGB images
                #target_rgb = np.array(loader.dataset.decode_target(target)).astype(np.uint8)
                pred_rgb = np.array(loader.dataset.decode_target(pred)).astype(np.uint8)

                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

                Image.fromarray(image).save('results/%d_image.png' % img_id)
                    
                #target_image = Image.fromarray(target_rgb).convert('RGB')
                #target_image.save('results/%d_target.png' % img_id)
                    
                pred_image = Image.fromarray(pred_rgb).convert('RGB')
                #pred_image.save('results/%d_pred.png' % img_id)

                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1
            
    
def validate(opts, model, loader, device, metrics, criterion, ret_samples_ids= None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    total_loss = 0.0  # Initialize total loss
    num_batches = 0

    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0


    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            #preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            
            outputs = torch.squeeze(outputs, dim=1)
            labels = labels.float()

            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item()  # Accumulate the total loss
            num_batches += 1

            probs = torch.sigmoid(outputs).detach()
            preds = (probs > 0.5).long().cpu().numpy()
            
            targets = labels.cpu().numpy()
            
            
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    # Decode the binary target and prediction masks to RGB images
                    target_rgb = np.array(loader.dataset.decode_target(target)).astype(np.uint8)
                    pred_rgb = np.array(loader.dataset.decode_target(pred)).astype(np.uint8)

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    
                    target_image = Image.fromarray(target_rgb).convert('RGB')
                    target_image.save('results/%d_target.png' % img_id)
                    
                    pred_image = Image.fromarray(pred_rgb).convert('RGB')
                    pred_image.save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        
        average_loss = total_loss / num_batches
        score = metrics.get_results()
    return score, ret_samples, average_loss


def smooth_labels(labels,smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing

def freeze_backbone_layers(model, freeze_until):
    layer_index = 0
    for layer in model.backbone.children():
        if layer_index < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
        layer_index += 1

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'custom':
        #opts.num_classes = 2   # Multi-class with cross-entropy
        opts.num_classes = 1   #Binary segmentation


    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    # Device Config
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    #test_loader = data.DataLoader(
        #test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    #freeze_backbone_layers(model, 4)
    
    # Set up metrics
    ## metrics = StreamSegMetrics(opts.num_classes)
    metrics = BinarySegMetrics()

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    
    elif opts.lr_policy == 'warmup':
        scheduler = utils.GradualWarmupLR(optimizer, multiplier=1, total_epoch=5, after_scheduler=None)
    
    elif opts.lr_policy == 'chained': # Combine warm up and step
        warmup_scheduler = utils.GradualWarmupLR(optimizer, multiplier=1, total_epoch=5, after_scheduler=None)
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, step_scheduler], milestones=[5])

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'binary_focal':
        criterion = utils.BinaryFocalLoss() 
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()  ##nn.BCELoss()
    elif opts.loss_type == 'binary_dice':
        criterion = DiceBCELoss()
    elif opts.loss_type == 'kl_dice_combined':
        criterion = CombinedLoss()
        

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # =========================================   Train Loop   =====================================================
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.generate_only:
        model.eval()
        print('Generating Predictions')
        generate(
            opts=opts, model=model, loader=val_loader, device=device)
        print('Everything saved to the results folder.')
        return

    if opts.test_only:
        model.eval()
        val_score, ret_samples, _ = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return
    
    if not os.path.exists('saved_models'):
            os.mkdir('saved_models')

    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = 5

    interval_loss = 0

    foreground_vals = []
    interval_vals = []
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, probs) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            probs = probs.to(device, dtype=torch.float32)
            #print(type(probs))
            #print(probs.shape)
            optimizer.zero_grad()
            #print("Max label value:", labels.max().item())
            outputs = model(images)
            

            # Dimension to BCE
            #outputs = torch.sigmoid(outputs)
            outputs = torch.squeeze(outputs, dim=1)
            student_probs = torch.sigmoid(outputs)
            #probs = torch.from_numpy(probs).unsqueeze(0)
            labels = labels.float()

            # Label smoothing
            labels = smooth_labels(labels, smoothing=0.1)

            loss = criterion(student_probs, probs, outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('saved_models/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples, current_val_loss = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion= DiceBCELoss(),
                    ret_samples_ids=vis_sample_id)
                
                print("Validation Loss: %f" % current_val_loss)
                print(metrics.to_str(val_score))
                foreground_vals.append(val_score['IoU Foreground'])
                interval_vals.append(cur_itrs)
                print(foreground_vals)
                print(interval_vals)
                save_ckpt('saved_models/kl_check.pth')
                print('saved here.')

                
                if val_score['IoU Foreground'] > best_score and opts.save_checkpoint:  # save best model
                    best_score = val_score['IoU Foreground']
                    #save_ckpt('saved_models/crf_sam_large_freetown_eastafrica_haiti_sam_threshold_97%s.pth' % ((val_score['IoU Foreground'])))
                    print('saved here.')

                    save_ckpt('saved_models/kl_check.pth')
                
                print("==========================================================")
                

#                 #================== Early stop =========================================
#                 if current_val_loss < best_val_loss:
#                     best_val_loss = current_val_loss
#                     no_improve_epochs = 0
#                     # Save best model
#                     save_ckpt('saved_models/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride))
#                 else:
#                     no_improve_epochs += 1

#                 if no_improve_epochs >= patience:
#                     print("Early stopping triggered after %d validations" % no_improve_epochs)
#                     break
#                 #=======================================================================

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Foreground Acc", cur_itrs, val_score['Foreground Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    #vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                    vis.vis_scalar("[Val] IoU Foreground", cur_itrs, val_score['IoU Foreground'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
