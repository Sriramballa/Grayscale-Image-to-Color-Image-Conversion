import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ColorizationCNN
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = ColorizationCNN().to(device)
model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
model.eval()

# Automatically pick the first image in the test folder
test_folder = "data/test"
print("Available test images:", os.listdir(test_folder))
img_name = input("Enter image name: ")
img_path = os.path.join(test_folder, img_name)


# Load and preprocess the image
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
gray = transform(img).unsqueeze(0).to(device)

# Predict color
with torch.no_grad():
    output = model(gray)

# Convert output tensor to image
output_image = transforms.ToPILImage()(output.squeeze().cpu())

# Display side by side
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Grayscale Input")
plt.imshow(img.convert("L"), cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Colorized Output")
plt.imshow(output_image)
plt.axis('off')

plt.show()

# Save the colorized image
output_image.save("output_color.jpg")
print(f"Colorized image saved as output_color.jpg")

from PIL import ImageEnhance
import numpy as np

# Convert grayscale image to RGB
gray_rgb = img.convert("RGB").resize((256, 256))

# Convert both to numpy
gray_np = np.array(gray_rgb).astype(float)
color_np = np.array(transforms.ToPILImage()(output)).astype(float)

# Blend: keep 80% grayscale structure, 20% color
final_np = 0.8 * gray_np + 0.2 * color_np
final_np = np.clip(final_np, 0, 255).astype(np.uint8)

output_image = Image.fromarray(final_np)




