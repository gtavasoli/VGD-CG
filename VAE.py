from dataset.dataset import CompositionDataset
from torch.utils.data import DataLoader
from VAE.model.vae import VAE_Conv
import torch
import pandas as pd 
import pickle
from pymatgen.core.composition import Composition
from utils.tool import *
from pandarallel import pandarallel
from tqdm.auto import tqdm
import os


def reconstruct_samples(model, data_loader=None):
    """
    Reconstructs samples using the given Variational Autoencoder (VAE) model and compares the 
    reconstructed samples with the original input.

    Args:
        model (torch.nn.Module): The trained VAE model used for reconstruction.
        data_loader (torch.utils.data.DataLoader, optional): DataLoader containing the input samples. 
            Defaults to None.

    Returns:
        None: Prints the reconstruction accuracy as the ratio of identical real and fake compositions.
    """

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store real (original) and fake (reconstructed) compositions
    comp_list = [[], []]

    # Create a progress bar for the reconstruction process
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (x, label) in loop:
        # Transfer input data to the appropriate device (e.g., GPU or CPU)
        x = x.to(device)
        # label = label.to(device)

        # Perform reconstruction using the model
        x_hat, _, _ = model(x)

        # Remove unnecessary dimensions for easier processing
        x = x.squeeze()
        x_hat = x_hat.squeeze()
        
        # Convert features to compositions for both original (x) and reconstructed (x_hat) data
        for i, d in enumerate([x, x_hat]):
            comp_list[i] += feature2composition(d, model.elements_list)
    
    # Combine real and fake compositions into a DataFrame for comparison
    data = pd.DataFrame({'real': comp_list[0], 'fake': comp_list[1]})
    
    # Calculate the number of samples where the real and fake compositions match
    same_num = sum(data['real'] == data['fake'])
    
    # Print reconstruction accuracy
    print("reconstruction: {}/{} {}".format(same_num, data.shape[0], same_num/data.shape[0]))
 

def generate_samples(
        model_path, 
        epochs=1, 
        batch_size=2500, 
        which_class=0, 
        device="cuda",
        is_icsd=False,
        is_semic=False
        ):
        """
        Helper function to iteratively generates valid compositions using a pre-trained Variational Autoencoder (VAE) model.

        This function decodes random noise vectors sampled from a standard normal distribution
        to generate synthetic material compositions. It incorporates optional class conditions
        (e.g., stability, ICSD, and semiconductor labels) and saves valid compositions to a CSV file.

        Args:
            model_path (str): Path to the saved VAE model.
            epochs (int, optional): Number of iterations for generating samples. Defaults to 1.
            batch_size (int, optional): Number of samples to generate in each batch. Defaults to 2500.
            which_class (int, optional): Index of the target class for conditional generation. Defaults to 0.
            device (str, optional): Device to use for computations ("cuda" or "cpu"). Defaults to "cuda".
            is_icsd (bool, optional): Condition to include ICSD label in generated samples. Defaults to False.
            is_semic (bool, optional): Condition to include semiconductor label in generated samples. Defaults to False.

        Returns:
            None: Saves the valid generated samples to `./output/VAE/generate_sample_cond.csv` and prints statistics.
        """

        # Initialize an empty list to store generated compositions
        comp_list = []
        
        # Load the pre-trained VAE model and set it to evaluation mode
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        
        # Generate synthetic compositions for the specified number of epochs
        for i in range(epochs):

            # Sample random latent vectors from a standard normal distribution
            fake_noise = torch.randn(batch_size, model.z_dim).float().to(device)
            # 条件编码

            # Apply conditional labels if the model supports them
            if model.class_label or model.icsd_label or model.semic_label:
                # print("*" * 100)

                # Create a label tensor with appropriate dimensions
                labels = torch.zeros(batch_size, model.n_class+model.icsd_label+model.semic_label, device=device)

                # Assign class-specific, ICSD, or semiconductor conditions
                if model.class_label:
                    labels[:, which_class] = 1
                
                if model.icsd_label:
                    labels[:, model.n_class] = is_icsd

                if model.semic_label:
                    labels[:, model.n_class+model.semic_label] = is_semic     
   
                # Decode latent vectors using the conditional labels
                fake_gen = model.decode(fake_noise, labels).cpu().squeeze().detach()
            else:
                # Decode latent vectors without conditional labels
                fake_gen = model.decode(fake_noise).cpu().squeeze().detach()
            
            # Convert generated features to material compositions
            comp_list += feature2composition(fake_gen, model.elements_list)

        # Create a DataFrame to store generated compositions
        data = pd.DataFrame({'composition': comp_list})
        data = data.dropna()    # Drop invalid compositions (None)
        print("Removed None values: {}/{}".format(data.shape[0], len(comp_list)))
        # data = data.drop_duplicates()
        # print("删除重复后：{}/{}".format(data.shape[0], len(comp_list)))
        

        pandarallel.initialize(progress_bar=False, nb_workers=48)

        # Filter compositions with element counts between 2 and 5
        data['element_num'] = data['composition'].parallel_apply(lambda comp: len(comp.elements))
        
        print(data['element_num'].value_counts())
     
        data = data[data['element_num']>1]
        data = data[data['element_num']<6]
        print("Compositions with 2-5 elements: {}/{}".format(data.shape[0], len(comp_list)))

        # Check chemical validity, charge neutrality, and electronegativity balance
        temp = data['composition'].parallel_apply(check_valid)
        data['charge neutrality'] = temp.parallel_apply(lambda comp: comp[0])
        data['electronegativity balance'] = temp.parallel_apply(lambda comp: comp[1])
        data['valid'] = temp.parallel_apply(lambda comp: comp[2])
        
        # Print statistics for validity checks
        print("charge neutrality: {}/{} {}".format(data['charge neutrality'].sum(), data.shape[0], 
                                        data['charge neutrality'].sum()/data.shape[0])
                                        )
        print("electronegativity balance: {}/{} {}".format(data['electronegativity balance'].sum(), 
                                        data.shape[0], data['electronegativity balance'].sum()/data.shape[0])
                                        )
        print("valid: {}/{} {}".format(data['valid'].sum(), data.shape[0], data['valid'].sum()/data.shape[0]))
        
        # Save valid compositions to a CSV file
        data['composition'] = data['composition'].parallel_apply(lambda comp: comp.formula)
        output_file = "./output/VAE/generate_sample_cond.csv"
        data.to_csv(output_file, index=False)

        print(f"Generated samples saved to {output_file}")
        
        

def loss_func(x, x_hat, mean, logvar, w_kl=1):
    """
    Computes the loss for the Variational Autoencoder (VAE) model.

    The loss function combines the weighted binary cross-entropy (BCE) reconstruction loss
    and the Kullback-Leibler (KL) divergence regularization term. The BCE loss ensures the
    accuracy of reconstruction, while the KL divergence promotes latent space regularization.

    Args:
        x (torch.Tensor): Original input tensor of shape (batch_size, ...).
        x_hat (torch.Tensor): Reconstructed output tensor of the same shape as `x`.
        mean (torch.Tensor): Mean vector of the latent space distribution.
        logvar (torch.Tensor): Log variance vector of the latent space distribution.
        w_kl (float, optional): Weight for the KL divergence term. Defaults to 1.

    Returns:
        torch.Tensor: The total loss value combining BCE and KL divergence terms.
    """

    # Flatten input and reconstructed tensors to compute BCE loss
    x = x.view(x.shape[0], -1)
    x_hat = x_hat.view(x_hat.shape[0], -1)

    # Assign weights: higher weight for non-zero values to balance the BCE loss
    weight = torch.where(x==0, 1/9, 8/9)

    # Define weighted binary cross-entropy loss
    bce = torch.nn.BCELoss(weight=weight, reduction='none')
    bce_loss = torch.mean(torch.sum(bce(x_hat, x), dim=1), dim=0)

    # Compute the KL divergence loss
    kl_loss = torch.mean(-0.5*torch.sum(1+logvar-mean.pow(2)-logvar.exp(), dim=1), dim=0)

    # Combine the BCE loss and weighted KL divergence loss
    total_loss = bce_loss + w_kl * kl_loss

    return total_loss


def train(config):
    """
    Trains a Variational Autoencoder (VAE) model for material composition generation.

    This function trains a VAE using the provided configuration parameters. It includes
    the setup of training and testing datasets, model initialization, and iterative
    optimization of the model's parameters using the specified loss function. The trained
    model is saved to a specified directory.

    Args:
        config (dict): Configuration dictionary containing training parameters. Expected keys include:
            - 'seed' (int): Random seed for reproducibility.
            - 'z_dim' (int): Dimensionality of the latent space.
            - 'w_kl' (float): Weight for the KL divergence term in the loss function.
            - 'batch_size' (int): Batch size for training.
            - 'semic_label' (bool): Flag for including semiconductor label in the dataset.
            - 'lr' (float): Learning rate for the optimizer.
            - 'epochs' (int): Number of training epochs.
            - 'n_class' (int): Number of classes for conditional training.
            - 'class_flag' (str): Class label for conditional training (e.g., "stability").
            - 'icsd_label' (bool): Flag for including ICSD labels.
            - 'model_name' (str): Name of the saved model file.
            - 'device' (str): Device to use for training ("cuda" or "cpu").

    Returns:
        None: The trained model is saved to the directory `./saved_model/VAE/`.
    """

    # Initialize training and testing datasets
    train_dataset = CompositionDataset(target_name="train.csv", n_class=config['n_class'], 
                                       class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                                       semic_label=config['semic_label']
                                       )
    test_dataset = CompositionDataset(target_name="test.csv", n_class=config['n_class'], 
                                      class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                                      semic_label=config['semic_label']
                                      )
    print("train length:", len(train_dataset), "test length:", len(test_dataset))

    # Create DataLoaders for efficient data processing
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24
                              ) 
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                             pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24
                             )
    
    # Save the DataLoaders for later use
    torch.save((train_loader, test_loader), "./data/train_test_loader.pt")

    # Initialize the VAE model
    model = VAE_Conv(train_dataset.data_size, config['z_dim'], train_dataset.elements_list, 
                     config['n_class'], class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                     semic_label=config['semic_label']
                     )
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    
    # Print model summary for verification
    device = config['device']
    print('Model architectures')
    model_summary(model)
    print("\n\n Starting VAE training\n\n")

    # Move the model to the specified device (e.g., GPU or CPU)
    model = model.to(device)
    model.train()

    # Training loop for the specified number of epochs
    epochs = config['epochs']
    for epoch in range(epochs): 
        
        # Use a progress bar to monitor training
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, label) in loop:
            
            # Transfer input data and labels to the specified device
            x = x.to(device)
            label = label.to(device)

            # Forward pass through the model
            x_hat, mean, logvar = model(x, label)

            # Compute the loss and perform backpropagation
            loss = loss_func(x, x_hat, mean, logvar, config['w_kl'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')

            # Update the progress bar with the current loss
            loop.set_postfix(loss=loss.item())

# if ((epoch+1) % 50 == 0):
    # print('\n计算重建损失率')
    # # print("train_loader:")
    # # reconstruct_samples(model, train_loader)
    # print("test_loader:")
    # reconstruct_samples(model, test_loader)
    
    # Save the trained model to the specified directory  
    save_path = os.path.join("./saved_model/VAE", config['model_name'])
    torch.save(model, save_path)
    print(f"\nFinished training. Model saved to {save_path}")
   
    
if __name__ == "__main__":
    """
    Main execution block for training and generating samples using the Variational Autoencoder (VAE).

    This block initializes the training configuration, sets up the environment, trains the VAE model,
    and generates synthetic material compositions. The results are saved to specified output files.
    """

    import time

    # Define the training configuration
    config = {
        'seed': 6,                   # Random seed for reproducibility
        'z_dim': 128,                # Dimensionality of the latent space
        'w_kl': 0.01,                # Weight for the KL divergence term in the loss function
        'batch_size': 5120,          # Batch size for training
        'semic_label': True,         # Include semiconductor labels
        'lr': 0.001,                 # Learning rate for the optimizer
        'epochs': 20,                # Number of training epochs
        'n_class': 2,                # Number of classes for conditional training
        'class_flag': "stability",   # Class label for conditional training
        'icsd_label': True,          # Include ICSD labels
        'model_name': 'VAE_Conv_cond.pt'  # Name of the saved model file
    }

    # Set the computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    print(f"Running on {device}")

    # Set random seed for reproducibility
    set_random_seed(config['seed'])
    config['device'] = device

    # Measure the start time for training
    time_1 = time.time()
    
    # Train the VAE model
    print("\nStarting the training process...")
    train(config)
    
    # Measure the time taken for training
    time_2 = time.time()
    print(f"Training completed in {time_2 - time_1:.2f} seconds.")

    # Generate synthetic samples using the trained model
    print("\nGenerating synthetic samples...")
    generate_samples(
        model_path=f"./saved_model/VAE/{config['model_name']}",  # Path to the trained model
        batch_size=10000,        # Number of samples per batch
        epochs=130,              # Number of iterations for sample generation
        is_icsd=True,            # Include ICSD condition in generated samples
        is_semic=True            # Include semiconductor condition in generated samples
    )

    # Measure the time taken for sample generation
    time_3 = time.time()
    print(f"Sample generation completed in {time_3 - time_2:.2f} seconds.")

    # Print total execution time
    print(f"Total execution time: {time_3 - time_1:.2f} seconds.")