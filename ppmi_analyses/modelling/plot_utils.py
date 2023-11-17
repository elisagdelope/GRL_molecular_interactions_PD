import seaborn as sns
from matplotlib import pyplot as plt

def training_plots (Nepochs, values_train, values_val, var_name, save_fig=False, name_file=None, path=None):
    array_epochs = range(1, Nepochs+1)

    # Plot and label the training and validation loss values
    plt.plot(array_epochs, values_train, label='Training ' + var_name)
    plt.plot(array_epochs, values_val, label='Validation ' + var_name)

    # Add in a title and axes labels
    plt.title('Training and Validation ' + var_name)
    plt.xlabel('Epochs')
    plt.ylabel(var_name)

    # Set the tick locations
    plt.xticks(range(0, Nepochs, (1 if Nepochs<40 else int(Nepochs/20))), fontsize=8)
    #plt.set_xticklabels(range(0, Nepochs, (1 if Nepochs<40 else int(Nepochs/20))), fontsize=15)
    #plt.tick_params(axis='both', which='minor', labelsize=12)

    # Display the plot
    plt.legend(loc='best')
    plt.show()
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()


def pca_plot2d(v1, v2, color_labels, save_fig=False, name_file=None, path=None):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=v1, y=v2,
        hue=color_labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.8
    )
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()

def pca_plot3d(v1, v2, v3, color_labels, save_fig=False, name_file=None, path=None):
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=v1,
        ys=v2,
        zs=v3,
        c=color_labels,
        cmap='viridis'
    )
    ax.set_xlabel('pca-v1')
    ax.set_ylabel('pca-v2')
    ax.set_zlabel('pca-v3')
    plt.show()
    if save_fig:
        plt.savefig(path + name_file)
    plt.close()