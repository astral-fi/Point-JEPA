from tokenizer import PointCloudTokenizer as Tokenizer
from dataset import DATASET_PATH, concatenate_h5_files, ModelNet40Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


data_train, label_train = concatenate_h5_files(DATASET_PATH, 5, 'train')
data_test, label_test = concatenate_h5_files(DATASET_PATH, 2, 'test')

dataset_train = ModelNet40Dataset(data_train, label_train, 1024)
dataset_train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataset_test = ModelNet40Dataset(data_test, label_test, 1024)
dataset_test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
print(f"Training dataset size: {len(dataset_train)}, Testing dataset size: {len(dataset_test)}")


class BaselineModel(nn.Module):
    def __init__(self, tokenizer):
        super(BaselineModel, self).__init__()
        self.tokenizer = tokenizer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, tokenizer.token_dim) * 0.02)

        self.linear_head = nn.Linear(tokenizer.token_dim, 40)  # Assuming 40 classes for ModelNet40

    def forward(self, x):
        tokens, positional_embeddings = self.tokenizer(x)
        combined_tokens = tokens + positional_embeddings  # Combine tokens with positional embeddings
        combined_tokens_shaped = combined_tokens.shape[0]
        cls = self.cls_token.expand(combined_tokens_shaped, -1, -1)  # Expand cls token to match batch size
        combined_tokens = torch.cat((cls, combined_tokens), dim=1)  # Concatenate cls token with the combined tokens
        x = self.encoder(combined_tokens)
        x = x[:, 0, :]  # Extract the cls token representation
        x = self.linear_head(x)
        return x


if __name__ == "__main__":
    tokenizer = Tokenizer(num_samples=64, k_neighbors=32, token_dim=128)
    tokenizer.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move the model to GPU if available
    model = BaselineModel(tokenizer)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move the model to GPU if available

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 50
    warmup_epochs = 5

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


    for epoch in range(EPOCHS):
        train_loss = 0.0
        correct = 0
        total = 0

        for points, labels in dataset_train_loader:

            points = points.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            logits = model(points)

            loss = criterion(logits, labels)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update the learning rate scheduler
            train_loss += loss.item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(dataset_train_loader):.4f}, Training Accuracy: {accuracy:.2f}%")

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0

            for points, labels in dataset_test_loader:
                points = points.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                logits = model(points)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")