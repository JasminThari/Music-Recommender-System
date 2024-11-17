import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

hex_color_list = ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"]

def plot_pca_variance(pca):
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Set the style for Seaborn
    sns.set_theme(style="whitegrid")

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(25, 9))

    # Variance explained plot
    sns.barplot(x=list(range(1, len(explained_variance) + 1)), y=explained_variance, ax=axs[0], color=hex_color_list[3])
    axs[0].set_title('Variance Explained by Each Principal Component', fontsize=16)
    axs[0].set_xlabel('Principal Component', fontsize=14)
    axs[0].set_ylabel('Variance Explained', fontsize=14)

    # Cumulative variance explained plot
    sns.lineplot(x=list(range(1, len(cumulative_variance) + 1)), y=cumulative_variance, ax=axs[1], marker='o', color= hex_color_list[1])
    axs[1].set_title('Cumulative Variance Explained', fontsize=16)
    axs[1].set_xlabel('Number of Principal Components', fontsize=14)
    axs[1].set_ylabel('Cumulative Variance Explained', fontsize=14)

    # Draw a horizontal line at 80% cumulative variance
    axs[1].axhline(y=0.8, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    
def plot_cluster_evaluation(scores: dict, num_clusters):

    wcss, silhouette_scores, davies_bouldin_indices = scores["wcss"], scores["silhouette_scores"], scores["davies_bouldin_indices"]

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Elbow Method plot with discrete x-axis
    sns.lineplot(x=range(1, num_clusters+1), y=wcss, marker='o', ax=ax1, color=hex_color_list[1])
    ax1.set_xticks(range(1, num_clusters+1))  # Ensure x-axis has all numbers as discrete values
    ax1.set_title('Elbow Method', fontsize=16)
    ax1.set_xlabel('Number of clusters', fontsize=14)
    ax1.set_ylabel('WCSS', fontsize=14)

    # Silhouette Score plot on the second subplot
    sns.lineplot(x=range(2, num_clusters+1), y=silhouette_scores, marker='o', color=hex_color_list[2], ax=ax2, label='Silhouette Score')
    ax2.set_title('Evaluation of Clustering Performance', fontsize=16)
    ax2.set_xlabel('Number of clusters', fontsize=14)
    ax2.set_ylabel('Silhouette Score', color='slategray', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='slategray')

    # Twin axis for Davies-Bouldin Index, with no grid
    ax2_twin = ax2.twinx()
    sns.lineplot(x=range(2, num_clusters+1), y=davies_bouldin_indices, marker='s', color=hex_color_list[4], ax=ax2_twin, label='Davies-Bouldin Index')
    ax2_twin.set_ylabel('Davies-Bouldin Index', color='darkslategray', fontsize=14)
    ax2_twin.tick_params(axis='y', labelcolor='darkslategray')
    ax2_twin.grid(False)  # Turn off grid for twin axis to reduce clutter

    # Add legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # Layout adjustment for clarity
    fig.tight_layout()
    plt.show()

def plot_cluster_distribution(df_pca):
    cluster_counts = df_pca['cluster'].value_counts().sort_index()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='inferno')

    # Add the count numbers on top of each bar
    for i, count in enumerate(cluster_counts.values):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12)

    # Set plot title and labels
    ax.set_title('Distributions of songs in each cluster', fontsize=16)
    ax.set_xlabel('Cluster', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_clustering(df_pca):
    # Set the style for Seaborn
    sns.set_theme(style="whitegrid")

    # Define the pairs of components to visualize
    first_row_pairs = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC1', 'PC4')]
    second_row_pairs = [('PC2', 'PC3'), ('PC2', 'PC4'), ('PC3', 'PC4')]

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loop over each pair for the first row
    for i, (x_pc, y_pc) in enumerate(first_row_pairs):
        sns.scatterplot(
            x=x_pc, y=y_pc, hue='cluster', data=df_pca,
            palette='inferno', ax=axes[0, i], s=60, edgecolor='w', alpha=0.7
        )
        axes[0, i].set_title(f'{x_pc} vs {y_pc}')
        axes[0, i].set_xlabel(x_pc)
        axes[0, i].set_ylabel(y_pc)

    # Loop over each pair for the second row
    for i, (x_pc, y_pc) in enumerate(second_row_pairs):
        sns.scatterplot(
            x=x_pc, y=y_pc, hue='cluster', data=df_pca,
            palette='inferno', ax=axes[1, i], s=60, edgecolor='w', alpha=0.7  # rocket, magma, coolwarm, viridis, inferno
        )
        axes[1, i].set_title(f'{x_pc} vs {y_pc}')
        axes[1, i].set_xlabel(x_pc)
        axes[1, i].set_ylabel(y_pc)

    # Adjust layout and show plot
    fig.tight_layout()
    plt.show()
        

def plot_loss(loss_history):
    # Convert loss history to a DataFrame
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(loss_history['train']) + 1),
        'Train Loss': loss_history['train'],
        'Validation Loss': loss_history['val']
    })

    # Melt the DataFrame for easier plotting with seaborn
    loss_df = loss_df.melt(id_vars='Epoch', value_vars=['Train Loss', 'Validation Loss'], 
                        var_name='Loss Type', value_name='Loss')

    # Plot the loss using seaborn
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=loss_df, x='Epoch', y='Loss', hue='Loss Type', marker='o', palette='inferno')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(title='Loss Type', fontsize=12)  
    plt.grid(True)
    plt.show()