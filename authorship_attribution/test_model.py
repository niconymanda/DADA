import torch
import config
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TesterAuthorshipAttribution:
    def __init__(self, model, repository_id, author_id_map, classification_model=None):
        self.model = model
        self.repository_id = repository_id
        self.author_id_map = author_id_map
        self.classification_model = classification_model if classification_model else None
        self.device = config.get_device()
        
    def test_abx_accuracy(self, test_dataloader):
        self.model.eval() 
        correct = 0
        total = 0

        with torch.no_grad(): 
            for batch in tqdm(test_dataloader):
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                anchor_labels = batch['label']

                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)

                dist_ax = F.pairwise_distance(anchor_embeddings, positive_embeddings)

                negative_labels = batch['negative_labels']
                negative_inputs_ids_list = batch['negative_input_ids']
                negative_attention_masks_list = batch['negative_attention_mask']

                for i in range(anchor_embeddings.size(0)): 
                    min_dist = dist_ax[i] 
                    correct_label = anchor_labels[i] 

                    for neg_idx in range(negative_inputs_ids_list.size(1)):  # N_negatives per anchor = #num_labels - 1
                        negative_input_ids = negative_inputs_ids_list[i, neg_idx].to(self.device)
                        negative_attention_mask = negative_attention_masks_list[i, neg_idx].to(self.device)
                        negative_embeddings = self.model(negative_input_ids.unsqueeze(0), negative_attention_mask.unsqueeze(0))
                        dist_bx = F.pairwise_distance(anchor_embeddings[i].unsqueeze(0), negative_embeddings)
                        if dist_bx < min_dist:
                            min_dist = dist_bx
                            correct_label = negative_labels[i][neg_idx]
                            # print(f"+: {dist_ax[i]}, _: {dist_bx}, anchor: {anchor_labels[i]}, negative: {negative_labels[i][neg_idx]}")
                            # print(f"Min dist: {min_dist}, possible label: {correct_label}")

                    if correct_label == anchor_labels[i]:
                        correct += 1
                    total += 1

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def test_classification(self, test_dataloader):
        self.classification_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['anchor_input_ids'].to(self.device)
                attention_mask = batch['anchor_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.classification_model(input_ids, attention_mask)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        return results

    def plot_tsne_for_authors(self, dataloader):
        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['anchor_input_ids'].to(self.device)
                attention_mask = batch['anchor_attention_mask'].to(self.device)
                labels = batch['label'].cpu().numpy() 

                embeddings = self.model(input_ids, attention_mask).cpu().numpy()

                all_embeddings.append(embeddings)
                all_labels.extend(labels)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        num_classes = len(set(all_labels))
        print(f"Number of classes: {num_classes}")
        tsne = TSNE(n_components=num_classes, perplexity=30, max_iter=3000, method='exact')
        tsne_results = tsne.fit_transform(all_embeddings)

        all_labels = np.array(all_labels)

        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=all_labels, palette=sns.color_palette("hsv", len(set(all_labels))),
            legend="full", alpha=0.8
        )

        # Modify legend to show author names
        handles, labels = scatter.get_legend_handles_labels()
        author_names = [self.author_id_map[int(label)] for label in labels]
        plt.legend(handles, author_names, title="Author Name")

        plt.title("t-SNE of Author Embeddings")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()
        plt.savefig(f"{self.repository_id}/t-SNE_plot_best_model.png")



