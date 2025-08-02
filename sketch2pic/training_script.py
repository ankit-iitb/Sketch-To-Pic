import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Import our custom modules
from models import (
    CUFSDataset, 
    UNetGenerator, 
    PatchGANDiscriminator,
    initialize_weights,
    setup_dataset_structure,
    generate_sample_images,
    load_and_generate
)

# Configuration
KAGGLE_INPUT_PATH = "/kaggle/input/cuhk-face-sketch-database-cufs"
KAGGLE_WORKING_PATH = "/kaggle/working"

class Sketch2PhotoTrainer:
    """Main trainer class for conditional GAN sketch-to-photo translation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {self.device}")
        
        self.setup_directories()
        self.setup_datasets()
        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()
        
    def setup_directories(self):
        """Create necessary directories for training outputs"""
        self.results_dir = os.path.join(KAGGLE_WORKING_PATH, "training_results")
        self.samples_dir = os.path.join(self.results_dir, "samples")
        self.models_dir = os.path.join(self.results_dir, "checkpoints")
        
        for directory in [self.results_dir, self.samples_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_datasets(self):
        """Initialize datasets and data loaders"""
        # Setup dataset structure
        self.data_dir = setup_dataset_structure(KAGGLE_INPUT_PATH, 
                                              os.path.join(KAGGLE_WORKING_PATH, "cufs_organized"))
        
        # Data augmentation and normalization
        self.transform = transforms.Compose([
            transforms.Resize((286, 286)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Create datasets
        self.train_dataset = CUFSDataset(self.data_dir, transform=self.transform, mode='train')
        self.val_dataset = CUFSDataset(self.data_dir, transform=self.transform, mode='val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
    
    def setup_models(self):
        """Initialize generator and discriminator models"""
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = PatchGANDiscriminator().to(self.device)
        
        # Initialize weights
        self.generator.apply(initialize_weights)
        self.discriminator.apply(initialize_weights)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def setup_optimizers(self):
        """Setup optimizers and learning rate schedulers"""
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config['lr_g'],
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['lr_d'],
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G, 
            T_max=self.config['epochs']
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D, 
            step_size=50, 
            gamma=0.5
        )
    
    def setup_losses(self):
        """Initialize loss functions"""
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.lambda_pixel = self.config['lambda_pixel']
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (sketches, photos) in enumerate(progress_bar):
            batch_size = sketches.size(0)
            sketches = sketches.to(self.device)
            photos = photos.to(self.device)
            
            # Train Generator
            self.optimizer_G.zero_grad()
            
            generated_photos = self.generator(sketches)
            
            # Adversarial loss (fool discriminator)
            pred_fake = self.discriminator(sketches, generated_photos)
            valid_labels = torch.ones_like(pred_fake, device=self.device) * 0.9  # Label smoothing
            loss_gan = self.adversarial_loss(pred_fake, valid_labels)
            
            # Reconstruction loss (L1)
            loss_pixel = self.reconstruction_loss(generated_photos, photos)
            
            # Combined generator loss
            loss_g = loss_gan + self.lambda_pixel * loss_pixel
            
            loss_g.backward()
            self.optimizer_G.step()
            
            # Train Discriminator
            self.optimizer_D.zero_grad()
            
            # Real images
            pred_real = self.discriminator(sketches, photos)
            real_labels = torch.ones_like(pred_real, device=self.device)
            loss_d_real = self.adversarial_loss(pred_real, real_labels)
            
            # Fake images
            pred_fake = self.discriminator(sketches, generated_photos.detach())
            fake_labels = torch.zeros_like(pred_fake, device=self.device) + 0.1  # Label smoothing
            loss_d_fake = self.adversarial_loss(pred_fake, fake_labels)
            
            # Combined discriminator loss
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            
            loss_d.backward()
            self.optimizer_D.step()
            
            # Update running losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{loss_g.item():.4f}',
                'D_Loss': f'{loss_d.item():.4f}',
                'Pixel': f'{loss_pixel.item():.4f}'
            })
        
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        
        return avg_g_loss, avg_d_loss
    
    def save_checkpoint(self, epoch, g_loss, d_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.models_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save generator separately for inference
        generator_path = os.path.join(self.models_dir, f"generator_epoch_{epoch+1}.pth")
        torch.save(self.generator.state_dict(), generator_path)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            # Train one epoch
            g_loss, d_loss = self.train_epoch(epoch)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
            
            # Generate sample images
            if (epoch + 1) % self.config['sample_interval'] == 0:
                generate_sample_images(
                    self.generator, 
                    self.val_loader, 
                    epoch + 1, 
                    self.samples_dir, 
                    self.device
                )
            
            # Save checkpoints
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(epoch, g_loss, d_loss)
        
        # Save final model
        final_generator_path = os.path.join(self.models_dir, "generator_final.pth")
        torch.save(self.generator.state_dict(), final_generator_path)
        
        print("Training completed!")
        return self.results_dir


def display_training_progress(results_dir):
    """Display sample images from training"""
    samples_dir = os.path.join(results_dir, "samples")
    
    if not os.path.exists(samples_dir):
        print("No training samples found!")
        return
    
    # Get latest sample images
    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.png')])
    
    if len(sample_files) > 0:
        # Show progression: first, middle, and last samples
        indices = [0, len(sample_files)//2, -1] if len(sample_files) > 2 else [0, -1]
        selected_samples = [sample_files[i] for i in indices]
        
        fig, axes = plt.subplots(len(selected_samples), 1, figsize=(15, 5*len(selected_samples)))
        if len(selected_samples) == 1:
            axes = [axes]
        
        for i, sample_file in enumerate(selected_samples):
            img_path = os.path.join(samples_dir, sample_file)
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Training Progress: {sample_file}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No sample images found!")


def main():
    """Main execution function"""
    print("=== Conditional GAN for Sketch-to-Photo Translation ===")
    
    # Check hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Training configuration
    config = {
        'epochs': 150,
        'batch_size': 8,
        'lr_g': 0.0001,
        'lr_d': 0.0002,
        'lambda_pixel': 100,
        'sample_interval': 10,
        'checkpoint_interval': 20
    }
    
    # Initialize and start training
    trainer = Sketch2PhotoTrainer(config)
    results_dir = trainer.train()
    
    # Display results
    print("\n=== Training Results ===")
    display_training_progress(results_dir)
    
    print(f"Training completed! Results saved in: {results_dir}")
    return results_dir


# For Kaggle environment - direct execution
if __name__ == "__main__":
    print("Starting Sketch2Photo cGAN training...")
    results_directory = main()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Models and samples saved in: {results_directory}")
    print("="*50)
