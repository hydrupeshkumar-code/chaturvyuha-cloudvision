import os
import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Fallback imports for package structure compatibility
try:
    from .generator import Generator
    from .discriminator import Discriminator
    from .dataset import Pix2PixDataset
except ImportError:
    from generator import Generator
    from discriminator import Discriminator
    from dataset import Pix2PixDataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate(generator, val_loader, criterion_pixelwise, device):
    """Evaluates the Generator on a held-out validation set."""
    generator.eval()
    running_l1_loss = 0.0
    with torch.no_grad():
        for imgs_cloudy, imgs_clear in val_loader:
            imgs_cloudy, imgs_clear = imgs_cloudy.to(device), imgs_clear.to(device)
            gen_clear = generator(imgs_cloudy)
            running_l1_loss += criterion_pixelwise(gen_clear, imgs_clear).item() * imgs_cloudy.size(0)
    generator.train()
    return running_l1_loss / max(1, len(val_loader.dataset))

def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix Conditional GAN for Cloud Removal")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-d", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    scaler = GradScaler() if device.type == "cuda" else None
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode="min", factor=0.5, patience=3, verbose=True)

    start_epoch, best_val_loss, early_stop_counter, history = 1, float('inf'), 0, []

    # Resume Logic
    if args.resume and os.path.exists(args.resume):
        try:
            checkpoint_g = torch.load(args.resume, map_location=device)
            generator.load_state_dict(checkpoint_g.get("model_state_dict", checkpoint_g))
            if "optimizer_state_dict" in checkpoint_g: optimizer_G.load_state_dict(checkpoint_g["optimizer_state_dict"])
            if "optimizer_D_state_dict" in checkpoint_g: optimizer_D.load_state_dict(checkpoint_g["optimizer_D_state_dict"])
            if "scheduler_state_dict" in checkpoint_g: scheduler_G.load_state_dict(checkpoint_g["scheduler_state_dict"])
            start_epoch = checkpoint_g.get("epoch", 0) + 1
            best_val_loss = checkpoint_g.get("val_l1", float('inf'))
            if os.path.exists(os.path.join(args.save_dir, "training_history.json")):
                with open(os.path.join(args.save_dir, "training_history.json"), 'r') as f: history = json.load(f)
        except Exception as e: logger.warning(f"Generator resume failed: {e}")

    if args.resume_d and os.path.exists(args.resume_d):
        try:
            checkpoint_d = torch.load(args.resume_d, map_location=device)
            discriminator.load_state_dict(checkpoint_d.get("model_state_dict", checkpoint_d))
            if "optimizer_state_dict" in checkpoint_d: optimizer_D.load_state_dict(checkpoint_d["optimizer_state_dict"])
        except Exception as e: logger.warning(f"Discriminator resume failed: {e}")

    train_dataset = Pix2PixDataset(os.path.join(args.data_dir, "train/cloudy"), os.path.join(args.data_dir, "train/clear"), transform=True)
    val_dataset = Pix2PixDataset(os.path.join(args.data_dir, "val/cloudy"), os.path.join(args.data_dir, "val/clear"), transform=False)
    
    if len(train_dataset) == 0: raise RuntimeError("Training dataset is empty")
    if len(val_dataset) == 0: raise RuntimeError("Validation dataset is empty")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=(len(train_dataset) >= args.batch_size), num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train(); discriminator.train()
        epoch_g, epoch_d = 0.0, 0.0
        
        for i, (imgs_cloudy, imgs_clear) in enumerate(train_loader):
            imgs_cloudy, imgs_clear = imgs_cloudy.to(device), imgs_clear.to(device)
            
            # Train G
            optimizer_G.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                gen_clear = generator(imgs_cloudy)
                pred_fake = discriminator(imgs_cloudy, gen_clear)
                loss_G = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) + (args.lambda_l1 * criterion_pixelwise(gen_clear, imgs_clear))
            
            if scaler:
                scaler.scale(loss_G).backward()
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler.step(optimizer_G)
            else:
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
            
            # Train D
            optimizer_D.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                pred_real = discriminator(imgs_cloudy, imgs_clear)
                pred_fake_detach = discriminator(imgs_cloudy, gen_clear.detach())
                loss_D = 0.5 * (criterion_GAN(pred_real, torch.ones_like(pred_real)) + 
                                criterion_GAN(pred_fake_detach, torch.zeros_like(pred_fake_detach)))
            if scaler:
                scaler.scale(loss_D).backward()
                scaler.unscale_(optimizer_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(optimizer_D)
                scaler.update() # Update scaler once per batch, after both optimizers have stepped
            else:
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()

            epoch_g += loss_G.item(); epoch_d += loss_D.item()
            
        avg_g, avg_d = epoch_g / max(1, len(train_loader)), epoch_d / max(1, len(train_loader))
        val_l1 = validate(generator, val_loader, criterion_pixelwise, device)
        scheduler_G.step(val_l1)
        
        logger.info(f"Epoch {epoch} | G={avg_g:.4f} | D={avg_d:.4f} | Val={val_l1:.4f}")
        
        writer.add_scalars("Loss/Combined", {"G": avg_g, "D": avg_d}, epoch)
        writer.add_scalar("Loss/Val_L1", val_l1, epoch)
        
        history.append({"epoch": epoch, "avg_g": avg_g, "avg_d": avg_d, "val_l1": val_l1})
        with open(os.path.join(args.save_dir, "training_history.json"), 'w') as f: json.dump(history, f, indent=4)
        
        # Save checkpoints
        torch.save({"epoch": epoch, "val_l1": val_l1, "best_val_l1": best_val_loss, "model_state_dict": generator.state_dict(), "optimizer_state_dict": optimizer_G.state_dict(), "scheduler_state_dict": scheduler_G.state_dict()}, os.path.join(args.save_dir, "generator_latest.pth"))
        torch.save({"epoch": epoch, "model_state_dict": discriminator.state_dict(), "optimizer_state_dict": optimizer_D.state_dict()}, os.path.join(args.save_dir, "discriminator_latest.pth"))
        
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "model_state_dict": generator.state_dict(), "optimizer_state_dict": optimizer_G.state_dict(), "scheduler_state_dict": scheduler_G.state_dict(), "val_l1": val_l1}, os.path.join(args.save_dir, f"generator_ep{epoch}.pth"))
            torch.save({"epoch": epoch, "model_state_dict": discriminator.state_dict(), "optimizer_state_dict": optimizer_D.state_dict()}, os.path.join(args.save_dir, f"discriminator_ep{epoch}.pth"))
        
        if val_l1 < best_val_loss:
            best_val_loss = val_l1; early_stop_counter = 0
            torch.save({"epoch": epoch, "val_l1": best_val_loss, "avg_g_loss": avg_g, "avg_d_loss": avg_d, "model_state_dict": generator.state_dict(), "optimizer_state_dict": optimizer_G.state_dict(), "optimizer_D_state_dict": optimizer_D.state_dict(), "scheduler_state_dict": scheduler_G.state_dict()}, os.path.join(args.save_dir, "generator_best.pth"))
            torch.save({"epoch": epoch, "model_state_dict": discriminator.state_dict(), "optimizer_state_dict": optimizer_D.state_dict()}, os.path.join(args.save_dir, "discriminator_best.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience: logger.info("Early stopping triggered."); break
        
    writer.close()

if __name__ == "__main__": main()