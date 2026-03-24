# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
To design and implement a Denoising Autoencoder using PyTorch that learns to reconstruct clean images from noisy input images using the MNIST dataset. The model should encode the input image into a compressed representation and then decode it to remove noise and reconstruct the original image. The performance of the model is evaluated by comparing the original, noisy, and denoised images.


<img width="750" height="234" alt="image" src="https://github.com/user-attachments/assets/722ca315-dcd7-4a02-8e8e-af7fb04b6ade" />


## DESIGN STEPS
### STEP 1: 
Examine the current setup for loading the MNIST dataset and the add_noise function to understand how input data is prepared and corrupted. This includes checking the transform applied to images.



### STEP 2: 
Review the DenoisingAutoencoder class definition, paying attention to the encoder and decoder layers, and their corresponding activation functions. Also, confirm the criterion (loss function) and optimizer used for training.



### STEP 3: 
Examine the train function to understand how the model is trained, including the epoch loop, batch processing, noise addition during training, forward pass, loss calculation, backpropagation, and optimizer step.


### STEP 4: 

Review the visualize_denoising function to understand how the model's performance is evaluated and visualized. Pay attention to how original, noisy, and denoised images are displayed side-by-side for comparison.

### STEP 5: 
Based on the output of the executed visualize_denoising function, analyze the effectiveness of the current autoencoder in removing noise. Identify patterns in denoised images and consider if the model is underfitting or overfitting.


### STEP 6: 
Summarize the current understanding of the denoising autoencoder's implementation and performance, and suggest potential next steps for improvement or further analysis based on the assessment.




## PROGRAM

### Name: Arun S

### Register Number:2122230023

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Initialize model
model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer= optim.Adam(model.parameters(),lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
  print("Name: Arun S               ")
  print("Register Number:   212224230023             ")
  model.train()
  for epoch in range(epochs):
    running_loss=0.0
    for images, _ in loader:
      images=images.to(device)
      noisy_images=add_noise(images).to(device)

      outputs=model(noisy_images)
      loss=criterion(outputs,images)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss+=loss.item()

    print(f"Epoch[{epoch+1}/{epoch}],Loss:{running_loss/len(loader):.4f}")


# Visualization function
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Arun S                  ")
    print("Register Number:     212224230023             ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



```

### OUTPUT

### Model Summary
<img width="728" height="418" alt="image" src="https://github.com/user-attachments/assets/5284df26-2a58-4d7f-885b-e4c5f94071c9" />


### Training loss
<img width="551" height="160" alt="image" src="https://github.com/user-attachments/assets/1f1f53cd-4179-4de3-a56f-a52673ca3414" />



## Original vs Noisy Vs Reconstructed Image

<img width="1401" height="511" alt="image" src="https://github.com/user-attachments/assets/a9b17fff-babb-46b6-8e8d-34582d824da8" />



## RESULT
The above code execute succesfully.
