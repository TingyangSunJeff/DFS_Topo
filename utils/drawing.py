# utils/drawing.py
import matplotlib.pyplot as plt
import networkx as nx

def draw_network(network, title="Network Graph", node_color='lightblue', edge_color='gray'):
    """Draw the network with nodes and edges."""
    pos = nx.spring_layout(network)
    nx.draw_networkx_nodes(network, pos, node_color=node_color)
    nx.draw_networkx_edges(network, pos, edge_color=edge_color)
    nx.draw_networkx_labels(network, pos)
    plt.title(title)
    plt.savefig('plot.png')

def draw_network_with_path(network, path, title="Network Path"):
    """Draw the network highlighting a specific path."""
    pos = nx.spring_layout(network)
    nx.draw_networkx(network, pos)
    path_edges = list(zip(path,path[1:]))
    nx.draw_networkx_edges(network, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title(title)
    plt.show()


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
    plt.savefig('plot_overlay.png')


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
    plt.savefig('plot.png')



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
    plt.savefig('plot_loss_acc_epoch.png')