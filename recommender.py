import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear

# Load the trained model
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['products'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

# Load data and model
data_path = 'data/product_data_PYG.pt'
model_path = 'models/amazon_best_model.pt'
reviews_path = 'data/organized_reviews.csv'
user_mapping_path = 'data/user_mapping.json'
rev_user_mapping_path = 'data/rev_user_mapping.json'

print("Loading data...")
data = torch.load(data_path, map_location=torch.device('cpu'))
device = 'cpu'
data = data.to(device)

print("Loading model...")
model = Model(hidden_channels=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

print("Loading reviews dataframe...")
reviews_df = pd.read_csv(reviews_path)

print("Loading user mappings...")
user_mapping = pd.read_json(user_mapping_path, typ='series').to_dict()
rev_user_mapping = pd.read_json(rev_user_mapping_path, typ='series').to_dict()

# Function to get the username from user_id
def get_username(user_id):
    if user_id not in reviews_df['user_id'].values:
        raise ValueError(f"User ID {user_id} not found in reviews_df")
    return reviews_df[reviews_df['user_id'] == user_id]['username'].iloc[0]

# Function to get product recommendations
def get_product_recommendations(model, data, user_id, total_products):
    user_idx = user_mapping[user_id]  # Get the embedding index for the user_id
    user_row = torch.tensor([user_idx] * total_products).to(device)
    all_product_ids = torch.arange(total_products).to(device)
    edge_label_index = torch.stack([user_row, all_product_ids], dim=0)
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index).cpu()
    top_five_indices = pred.topk(5).indices.numpy()  # Ensure indices are integers for indexing
    recommendations = []
    for idx in top_five_indices:
        idx = int(idx)  # Convert to integer for indexing
        product_id = reviews_df.iloc[idx]['product_id']
        category = reviews_df.iloc[idx]['category']
        subcategory = reviews_df.iloc[idx]['subcategory']
        recommendations.append((product_id, category, subcategory))
    return recommendations

# Function to get and print recommendations for a given user
def get_recommendations(user_id):
    try:
        user_id = str(user_id)
        username = get_username(user_id)
        recommendations = get_product_recommendations(model, data, user_id, data['products'].x.shape[0])
        return f"Recommendations for {username} (User ID: {user_id}):", recommendations
    except Exception as e:
        return f"Error: {str(e)}", []

if __name__ == "__main__":
    # For testing the recommendation functionality
    user_id = 'A314APAWYQFKBJ'  # Example user ID
    recommendations_title, recommendations = get_recommendations(user_id)
    print(recommendations_title)
    print(recommendations)
