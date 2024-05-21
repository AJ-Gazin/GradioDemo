import gradio as gr
import plotly.express as px
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from recommender import get_recommendations

# Loading pre-trained product embeddings for visualization
product_embeddings_path = 'data/product_embeddings.pt'
product_emb = torch.load(product_embeddings_path, map_location=torch.device('cpu'))

# Loading pre-trained user embeddings for visualization
user_embeddings_path = 'data/user_embeddings.pt'
user_emb = torch.load(user_embeddings_path, map_location=torch.device('cpu'))

# Loading the reviews dataframe for visualization purposes
reviews_df = pd.read_csv('data/organized_reviews.csv')

# Loading the training and validation loss data
loss_data_path = 'data/loss_data.csv'
loss_df = pd.read_csv(loss_data_path)
loss_df.columns = ['Epoch', 'Training Loss', 'Validation Loss']

# Creating a user dataframe by extracting unique user IDs and usernames
user_df = reviews_df[['user_id', 'username']].drop_duplicates()


# Function to perform dimensionality reduction on embeddings
# This function reduces the high-dimensional embeddings to a lower-dimensional space for visualization
def reduce_dimensions(embeddings, method, n_components=3):
    # Selecting the appropriate dimensionality reduction technique based on the specified method
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    else:
        # Performing initial PCA to reduce dimensionality before applying t-SNE or UMAP
        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(embeddings)
        reducer = TSNE(n_components=n_components) if method == "TSNE" else umap.UMAP(n_components=n_components)
        embeddings = reduced_embeddings

    # Applying the selected dimensionality reduction technique to the embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Assigning appropriate column names based on the dimensionality reduction method
    columns = ['PC1', 'PC2', 'PC3'] if method == "PCA" else ['TSNE1', 'TSNE2', 'TSNE3'] if method == "TSNE" else ['UMAP1', 'UMAP2', 'UMAP3']
    
    return reduced_embeddings, columns

# Function to visualize embeddings using interactive 3D scatter plots
# This function creates an interactive plot to explore the embeddings in a three-dimensional space
def visualize_embeddings(embeddings, df, method, is_product=True):
    reduced_embeddings, columns = reduce_dimensions(embeddings, method)
    df_reduced = pd.DataFrame(reduced_embeddings, columns=columns)
    
    if is_product:
        # Adding product-related information to the dataframe for hover interactions
        df_reduced['product_id'] = df['product_id']
        df_reduced['category'] = df['category']
        fig = px.scatter_3d(df_reduced, x=columns[0], y=columns[1], z=columns[2], color='category', hover_data=['product_id'], opacity=0.9)
    else:
        # Adding user-related information to the dataframe for hover interactions
        df_reduced['user_id'] = df['user_id']
        df_reduced['username'] = df['username']
        fig = px.scatter_3d(df_reduced, x=columns[0], y=columns[1], z=columns[2], hover_data=['user_id', 'username'], opacity=0.9)
    
    return fig

# Function to visualize product embeddings
# This function specifically visualizes the product embeddings using the selected dimensionality reduction method
def visualize_product_embeddings(method):
    return visualize_embeddings(product_emb.cpu().numpy(), reviews_df, method)

# Function to visualize user embeddings
# This function specifically visualizes the user embeddings using the selected dimensionality reduction method
def visualize_user_embeddings(method):
    return visualize_embeddings(user_emb.cpu().numpy(), user_df, method, is_product=False)

# Function to visualize training and validation loss
# This function creates a line plot to visualize the model's training and validation loss over epochs
def visualize_loss():
    fig = px.line(loss_df, x='Epoch', y=['Training Loss', 'Validation Loss'], labels={
        'Epoch': 'Epoch',
        'value': 'Loss',
        'variable': 'Loss Type'
    })
    fig.update_layout(title='Training and Validation Loss', legend_title='Loss Type')
    return fig

# Function to generate product recommendations for a given username
# This function retrieves the user ID based on the provided username and generates personalized product recommendations
def recommend(username, method):
    user_id = user_df[user_df['username'] == username]['user_id'].values[0]
    recommendations_title, recommendations = get_recommendations(user_id)
    recommendations_list = [[rec[0], rec[1], rec[2]] for rec in recommendations]
    return recommendations_title, recommendations_list


# Sampling a subset of usernames for the dropdown menu
sample_usernames = user_df['username'].sample(5, random_state=42).tolist()

# Creating the Gradio interface for the recommendation system
with gr.Blocks() as demo:
    gr.Markdown("# Amazon Product Recommendation System")

    with gr.Column():
        username_input = gr.Dropdown(label="Select Username", choices=sample_usernames, value=sample_usernames[0])
        recommendations_output = gr.Textbox(label="Recommendations")
        recommendations_list = gr.Dataframe(headers=["Product ID", "Category", "Subcategory"])
        recommend_button = gr.Button("Get Recommendations")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Product Embeddings Visualization")
            method_input_product = gr.Dropdown(label="Visualization Method", choices=["PCA", "TSNE", "UMAP"], value="PCA")
            embeddings_plot_product = gr.Plot(value=visualize_product_embeddings("PCA"))
        with gr.Column():
            gr.Markdown("### User Embeddings Visualization")
            method_input_user = gr.Dropdown(label="Visualization Method", choices=["PCA", "TSNE", "UMAP"], value="PCA")
            embeddings_plot_user = gr.Plot(value=visualize_user_embeddings("PCA"))

    gr.Markdown("### Training and Validation Loss")
    loss_plot = gr.Plot(value=visualize_loss())

    # Event triggers and their corresponding actions
    recommend_button.click(recommend, inputs=[username_input], outputs=[recommendations_output, recommendations_list])
    method_input_product.change(visualize_product_embeddings, inputs=[method_input_product], outputs=[embeddings_plot_product])
    method_input_user.change(visualize_user_embeddings, inputs=[method_input_user], outputs=[embeddings_plot_user])

# Running the Gradio interface
if __name__ == "__main__":
    demo.launch()