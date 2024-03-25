import pickle
from utils import plot_acc_loss_over_epochs, plot_acc_loss_for_different_times
# Load from file
with open('model_metrics.pkl', 'rb') as file:
    loaded_loss_history, loaded_val_accuracies = pickle.load(file)

plot_acc_loss_over_epochs(loaded_loss_history, loaded_val_accuracies)

with open('tau_results.pkl', 'rb') as file:
    time_per_epoch_list = pickle.load(file)
print(time_per_epoch_list)
labels = ['tau_random', 'tau_ring', "tau_baseline"]  
plot_acc_loss_for_different_times(loaded_loss_history, loaded_val_accuracies, time_per_epoch_list, labels)