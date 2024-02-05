
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as Image

class FeatureExtractor:
    
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        # Remove fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

        # Set model to evaluation mode
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_embeddings(self, image):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        # Convert grayscale to RGB if image has only 1 channel
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations
        input_image = self.transform(image)
        input_image = input_image.unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            features = self.model(input_image)

        # Flatten the features
        embeddings = features.view(features.size(0), -1)

        return embeddings.numpy()
    
    def compute_similarity(self, embeddings1, embeddings2):
        # Convert NumPy arrays to PyTorch tensors
        embeddings1_tensor = torch.from_numpy(embeddings1)
        embeddings2_tensor = torch.from_numpy(embeddings2)

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(embeddings1_tensor, embeddings2_tensor)
        return cosine_sim.item()