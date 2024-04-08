# utils/drawing.py
import matplotlib.pyplot as plt
import networkx as nx
import os

def draw_network(network, title="Network Graph", node_color='lightblue', edge_color='gray'):
    """Draw the network with nodes and edges."""
    pos = nx.spring_layout(network)
    nx.draw_networkx_nodes(network, pos, node_color=node_color)
    nx.draw_networkx_edges(network, pos, edge_color=edge_color)
    nx.draw_networkx_labels(network, pos)
    plt.title(title)
    save_path = os.path.join(os.getcwd(), 'graph_result', 'plot_underlay.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

def draw_fully_connected_overlay_network(overlay, overlay_nodes, activated_links):
    # Setting up the plot
    plt.figure(figsize=(10, 8))

    # Position nodes using a spring layout
    pos = nx.spring_layout(overlay)

    # Draw overlay network
    nx.draw_networkx_nodes(overlay, pos, nodelist=overlay_nodes, node_color='yellow', label='Overlay Nodes')
    nx.draw_networkx_edges(overlay, pos, edgelist=overlay.edges(), edge_color='blue', label='Overlay Edges', style='dotted')

    # Draw activated overlay links
    nx.draw_networkx_edges(overlay, pos, edgelist=activated_links, edge_color='red', label='Activated Overlay Links', style='solid')

    # Draw node labels
    nx.draw_networkx_labels(overlay, pos)

    # Set plot title and legends
    plt.title('Fully Connected Overlay Network with Activated Links')
    plt.legend()

    # Display the plot
    save_path = os.path.join(os.getcwd(), 'graph_result', 'plot_overlay.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def draw_underlay_network_with_mst(underlay, mst):
    # Setting up the plot
    plt.figure(figsize=(10, 8))

    # Position nodes using a spring layout
    pos = nx.spring_layout(underlay)

    # Draw underlay network
    nx.draw_networkx_nodes(underlay, pos, node_color='lightblue', label='Underlay Nodes')
    nx.draw_networkx_edges(underlay, pos, alpha=0.3, edge_color='gray', label='Underlay Edges')

    # Draw Minimum Spanning Tree
    mst_edges = mst.edges()
    nx.draw_networkx_edges(mst, pos, edgelist=mst_edges, edge_color='green', label='MST Edges', style='dashdot')

    # Draw node labels
    nx.draw_networkx_labels(underlay, pos)

    # Set plot title and legends
    plt.title('Underlay Network with MST')
    plt.legend()

    # Display the plot
    save_path = os.path.join(os.getcwd(), 'graph_result', 'plot_underlay.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)



def plot_acc_loss_over_epochs(lost_hitory, val_accuracies):

    epochs = range(1, len(lost_hitory) + 1)

    plt.figure(figsize=(10, 5))

    # Plotting loss history
    plt.subplot(1, 2, 1)
    plt.plot(epochs, lost_hitory, label='Loss')
    plt.title('Global Average Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
     # Display the plot
    # Construct the save path dynamically and save the figure
    save_path = os.path.join(os.getcwd(), 'graph_result', 'plot_overlay.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def plot_acc_loss_for_different_times(loss_history, val_accuracies, time_per_epoch_list, labels):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(14, 6))

    # Plotting loss for different time per epoch
    plt.subplot(1, 2, 1)
    for time_per_epoch, label in zip(time_per_epoch_list, labels):
        cumulative_time_slots = [time_per_epoch * epoch for epoch in epochs]
        plt.plot(cumulative_time_slots, loss_history, label=f'Loss - {label}')
    plt.title('Global Average Loss over Cumulative Time')
    plt.xlabel('Cumulative Time')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation accuracy for different time per epoch
    plt.subplot(1, 2, 2)
    for time_per_epoch, label in zip(time_per_epoch_list, labels):
        cumulative_time_slots = [time_per_epoch * epoch for epoch in epochs]
        plt.plot(cumulative_time_slots, val_accuracies, label=f'Accuracy - {label}', marker='o')
    plt.title('Validation Accuracy over Cumulative Time')
    plt.xlabel('Cumulative Time')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plot_loss_acc_different_times.png')


def plot_degree_distribution(node_degrees):
    """
    Plot the degree distribution of a network.

    Parameters:
    - node_degrees (dict): A dictionary where keys are node indices and values are the degrees of those nodes.
    """
    # Extract degree values from the dictionary and sort them
    degrees = list(node_degrees.values())
    
    # Create a frequency distribution of degrees
    degree_counts = {}
    for degree in degrees:
        if degree in degree_counts:
            degree_counts[degree] += 1
        else:
            degree_counts[degree] = 1
            
    # Prepare data for plotting
    degrees = list(degree_counts.keys())
    counts = list(degree_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, counts, color='skyblue')
    
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.xticks(degrees)  # Set x-ticks to be the degrees, for better visualization
    plt.grid(axis='y', linestyle='--')
    
    save_path = os.path.join(os.getcwd(), 'graph_result', 'degree.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)