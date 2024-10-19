import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(model, test_dataloader, device):
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for batch in tqdm(test_dataloader):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            anchor_labels = batch['label']
            
            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model(positive_input_ids, positive_attention_mask)
            
            dist_ax = F.pairwise_distance(anchor_embeddings, positive_embeddings)
            
            negative_labels = batch['negative_labels']
            negative_inputs_ids_list = batch['negative_input_ids']
            negative_attention_masks_list = batch['negative_attention_mask']
            
            for i in range(anchor_embeddings.size(0)): 
                min_dist = dist_ax[i] 
                correct_label = anchor_labels[i] 

                for neg_idx in range(negative_inputs_ids_list.size(1)):  # N_negatives per anchor = #num_labels - 1
                    negative_input_ids = negative_inputs_ids_list[i, neg_idx].to(device)
                    negative_attention_mask = negative_attention_masks_list[i, neg_idx].to(device)
                    negative_embeddings = model(negative_input_ids.unsqueeze(0), negative_attention_mask.unsqueeze(0))
                    dist_bx = F.pairwise_distance(anchor_embeddings[i].unsqueeze(0), negative_embeddings)
                    print(f"+: {dist_ax[i]}, _: {dist_bx}, anchor: {anchor_labels[i]}, negative: {negative_labels[i][neg_idx]}")
                    if dist_bx < min_dist:
                        min_dist = dist_bx
                        correct_label = negative_labels[i][neg_idx]
                        print(f"Min dist: {min_dist}, possible label: {correct_label}")
                        

                if correct_label == anchor_labels[i]:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def plot_tsne_for_authors(model, dataloader, device, repository):
    model.eval()
    
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['anchor_input_ids'].to(device)
            attention_mask = batch['anchor_attention_mask'].to(device)
            labels = batch['label'].cpu().numpy() 

            embeddings = model(input_ids, attention_mask).cpu().numpy()

            all_embeddings.append(embeddings)
            all_labels.extend(labels)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    num_classes = len(set(all_labels))
    print(f"Number of classes: {num_classes}")
    tsne = TSNE(n_components=num_classes, perplexity=30, max_iter=3000, method='exact')
    tsne_results = tsne.fit_transform(all_embeddings)

    all_labels = np.array(all_labels)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_labels, palette=sns.color_palette("hsv", len(set(all_labels))),
        legend="full", alpha=0.8
    )
    
    plt.title("t-SNE of Author Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Author ID")
    plt.show()
    plt.savefig(f"{repository}/t-SNE_plot.png")



