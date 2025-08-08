import imageio.v2 as iio
import numpy as np
import torch
import extract_foundation
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

foundation = extract_foundation.load_Foundation().to(device)
foundation.eval()

image = np.float32(iio.imread('./dataset/test/test3.jpg')) / 255.

image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


image = image.to(device)

with torch.no_grad():
    output = foundation(image)

for i in range(32):
    plt.imshow(output[0, i].detach().cpu().numpy())
    plt.show()