import os
import pandas as pd
import numpy as np
import shutil
from collections import Counter
import argparse
import ast
import re
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon, LineString
from matplotlib.patches import PathPatch
from matplotlib.patches import Patch
from matplotlib.path import Path
from utils import *

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('dir_path', help='Directory with wandb sweeps results')
args = parser.parse_args()
DIR = args.dir_path #"../results/wandb/DENOVO_GCNN_3p_uw_20231021_undersamping"
DIR_RESULTS = DIR + "/cv-results"
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)
annotation_file = "../data/chemical_annotation.tsv"
annotation = pd.read_csv(annotation_file, sep='\t', low_memory=False)
annotation = annotation[["ANALYSIS_ID", "CHEMICAL_NAME", "SUPER_PATHWAY", "SUB_PATHWAY"]]
annotation.drop_duplicates(inplace=True)

# Define the filename pattern to match
val_pattern = "_val_performance.csv"
# Get a list of all files in the folder that match the pattern
val_files = [f for f in os.listdir(DIR) if f.endswith(val_pattern)]
# select first 130 sweeps in case more were generated
val_files = sorted(val_files, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
val_files = val_files[:130]

# Initialize an empty dictionary to store the lowest loss values
lowest_losses = {}
# Iterate over the files
for file in val_files:
    # Read the file into a pandas DataFrame
    df = pd.read_csv(os.path.join(DIR, file))
    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the current row's loss value
        loss = row["Loss"]
        # Check if we already have a lowest loss value for this row
        if index not in lowest_losses or loss < lowest_losses[index]["Loss"]:
            # If not, update the lowest_losses dictionary with the new lowest loss value and the file name
            lowest_losses[index] = {"Loss": loss, "Sweep": file.replace(val_pattern, "")}
            print(index, loss, file)
for index, result in lowest_losses.items():
    print(f"For fold {index}, the sweep with the lowest loss value is {result['Sweep']}")

columns = ['Fold', 'AUC', 'Accuracy', 'Recall', 'Specificity', 'F1', 'N_epoch']
cvperformance_df = pd.DataFrame(columns=columns)
features_df = pd.DataFrame(columns=['Fold', 'Relevant_Nodes',	'Relevant_Edges_source','Relevant_Edges_dest'])

# Loop over each winner file in the lowest_losses dictionary
# retrieve test performance, relevant features, and graph images
for index, min_loss in lowest_losses.items():
    # test performance
    test_file = [f for f in os.listdir(DIR) if (f.startswith(min_loss["Sweep"]) and f.endswith("test_performance.csv"))][0]
    test_df = pd.read_csv(os.path.join(DIR, test_file))
    cvperformance_df = cvperformance_df.append(test_df.loc[index,columns], ignore_index=True)
    # relevant features
    features_file = [f for f in os.listdir(DIR) if (f.startswith(min_loss["Sweep"]) and f.endswith("features_track.csv"))][0]
    features = pd.read_csv(os.path.join(DIR, features_file))
    features_df = features_df.append(features.loc[index,['Fold', 'Relevant_Nodes',	'Relevant_Edges_source','Relevant_Edges_dest']], ignore_index=True)
mean_row = cvperformance_df.mean()
std_row = cvperformance_df.std()
# append the mean and std rows to the DataFrame
cvperformance_df = cvperformance_df.append(mean_row.rename('average'))
cvperformance_df = cvperformance_df.append(std_row.rename('std'))
cvperformance_df[['Fold', 'N_epoch']] = cvperformance_df[['Fold', 'N_epoch']].astype(int)
cvperformance_df.loc[['average', 'std'],['Fold', 'N_epoch']] = "NA"
cvperformance_df['Fold'] = cvperformance_df.index


# features (nodes): Relevant nodes (top-20) by frequency across cv folds -------------------------------------------------------------
n_nodes = 20
selected_nodes = [
    features[:n_nodes]
    for features in features_df['Relevant_Nodes'].apply(ast.literal_eval)]
# Flatten the list of selected nodes
flat_selected_nodes = [node for sublist in selected_nodes for node in sublist]

# Count the occurrences of each feature
node_counts = Counter(flat_selected_nodes) # all_nodes
nodes_df = pd.DataFrame(node_counts.items(), columns=['Feature', 'Count'])
nodes_df.sort_values(by='Count', ascending=False, inplace=True)
nodes_df = nodes_df.merge(annotation, left_on="Feature", right_on="ANALYSIS_ID", how="left")
nodes_df.drop("ANALYSIS_ID", axis=1, inplace=True)


# features (edges) -------------------------------------------------------------
n = 100
origin_edges = [
    features[:n]
    for features in features_df['Relevant_Edges_source'].apply(ast.literal_eval)]
flat_origin_edges = [node for sublist in origin_edges for node in sublist]
dest_edges = [
    features[:n]
    for features in features_df['Relevant_Edges_dest'].apply(ast.literal_eval)]
flat_dest_edges = [node for sublist in dest_edges for node in sublist]
edges_df = pd.DataFrame({'source': flat_origin_edges, 'dest': flat_dest_edges})

# relevant edges ocurring across folds
edges_df = edges_df.groupby(['source', 'dest']).size().reset_index(name='Occurrence').sort_values(by='Occurrence', ascending=False)
edges_df = edges_df.merge(annotation, left_on="source", right_on="ANALYSIS_ID", how="left")
edges_df.drop(["ANALYSIS_ID", "SUPER_PATHWAY"], axis=1, inplace=True)
edges_df.rename(columns=dict(zip(['CHEMICAL_NAME', 'SUB_PATHWAY'], [col + '_source' for col in ['CHEMICAL_NAME', 'SUB_PATHWAY']])), inplace=True)
edges_df = edges_df.merge(annotation, left_on="dest", right_on="ANALYSIS_ID", how="left")
edges_df.drop(["ANALYSIS_ID", "SUPER_PATHWAY"], axis=1, inplace=True)
edges_df.rename(columns=dict(zip(['CHEMICAL_NAME', 'SUB_PATHWAY'], [col + '_dest' for col in ['CHEMICAL_NAME', 'SUB_PATHWAY']])), inplace=True)

# explore the subgraph of relevant edges
G = nx.from_pandas_edgelist(edges_df, 'source', 'dest', edge_attr='Occurrence')
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G_components = [len(c) for c in Gcc]
print(len(G_components))
print(G_components[0:20])
print(np.unique(G_components, return_counts=True))

# plot largest connected component of relevant edges with labels for high degree nodes
G_connected = G.subgraph(Gcc[0])
plot_connected_graph(G_connected, annotation, d=8, fig_save=True, labels=True, path=DIR_RESULTS, plotname="cv_relevantedges_labels")
plot_connected_graph(G_connected, d=8, fig_save=True, labels=False, path=DIR_RESULTS, plotname="cv_relevantedges")

# nodes with highest degree among the relevant edges
degrees = dict(G.degree())
node_degree = pd.DataFrame(list(degrees.items()), columns=['Node', 'Degree'])
node_degree = node_degree.sort_values(by='Degree', ascending=False)
node_degree = node_degree.merge(annotation, left_on="Node", right_on="ANALYSIS_ID", how="left")
node_degree.drop("ANALYSIS_ID", axis=1, inplace=True)

# Highly connected nodes (degree) for edges being relevant in > k folds
k = 2
duplicate_edges = edges_df[edges_df['Occurrence'] >= k]
all_nodes = pd.concat([duplicate_edges['source'], duplicate_edges['dest']], ignore_index=True)
node_counts = all_nodes.value_counts()
node_counts = pd.DataFrame({'Node': node_counts.index, 'Degree (duplicated)': node_counts.values})
node_counts = node_counts.merge(annotation, left_on="Node", right_on="ANALYSIS_ID", how="left")
node_counts.drop("ANALYSIS_ID", axis=1, inplace=True)


# plot largest connected components (>=min_nodes) of duplicated relevant edges 
G = nx.from_pandas_edgelist(duplicate_edges, 'source', 'dest', edge_attr='Occurrence')
plot_largest_connected_components(G, labels=True, min_nodes=4, d=None, annotation=annotation, fig_save=True, path=DIR_RESULTS, plotname="cv_overlappingedges_components")
plot_largest_connected_components2(G, labels=True, min_nodes=4, d=None, annotation=annotation, fig_save=True, path=DIR_RESULTS, plotname="cv_overlappingedges_main_components")


# export results
cvperformance_df.to_csv(DIR_RESULTS + "/cv_test_results.csv", index=False)
nodes_df.to_csv(DIR_RESULTS + "/cv_relevantnodes.csv", index=False)
node_degree.to_csv(DIR_RESULTS + "/cv_relevantedges_nodesdegree.csv", index=False)
node_counts.to_csv(DIR_RESULTS + "/cv_overlappingedges_nodes.csv", index=False)
edges_df.to_csv(DIR_RESULTS + "/cv_relevantedges.csv", index=False)
duplicate_edges.to_csv(DIR_RESULTS + "/cv_overlappingedges.csv", index=False)



