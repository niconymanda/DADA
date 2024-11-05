import torch
import config
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

class TesterAuthorshipAttribution:
    """
    A class used to test authorship attribution models.
    Attributes
    ----------
    model : torch.nn.Module
        The authorship attribution model to be tested.
    repository_id : str
        The identifier for the repository where the logs and model will be saved.
    author_id_map : dict
        A mapping from author IDs to author names.
    classification_model : torch.nn.Module, optional
        The classification model to be used for testing (default is None).
    device : torch.device
        The device to run the model on (e.g., 'cpu' or 'cuda').
    """
    
    def __init__(self, model, repository_id, author_id_map, 
                 classification_model=None, 
                 distance_function='l2'):
        self.model = model
        self.repository_id = repository_id
        self.author_id_map = author_id_map
        self.classification_model = classification_model if classification_model else None
        self.device = config.get_device()
        self.all_cosine_distances_positive = []
        self.all_cosine_distances_negative = []

        if distance_function == 'l2':
            self.distance_function = torch.pairwise_distance
        elif distance_function == 'cosine':
            self.distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        
    def test_abx_accuracy(self, test_dataloader):
        """
        Evaluate the ABX accuracy of the model using the provided test dataloader (can be bona-fide and spoofed).
        Test the model by comparing the distance between the anchor and positive examples with the distance between the anchor and minimum of negative examples.
        Args:
            test_dataloader (DataLoader): A DataLoader object that provides batches of test data. Each batch should be a dictionary containing:
                - 'anchor_input_ids': Tensor of input IDs for anchor examples.
                - 'anchor_attention_mask': Tensor of attention masks for anchor examples.
                - 'positive_input_ids': Tensor of input IDs for positive examples.
                - 'positive_attention_mask': Tensor of attention masks for positive examples.
                - 'label': Tensor of labels for anchor examples.
                - 'negative_labels': Tensor of labels for negative examples.
                - 'negative_input_ids': Tensor of input IDs for negative examples.
                - 'negative_attention_mask': Tensor of attention masks for negative examples.
        Returns:
            float: The accuracy of the model on the test set.
        """
        
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

                # Anchor-Postive distances
                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)
                dist_ax = self.distance_function(anchor_embeddings, positive_embeddings)
                
                cosine_similarity_pos = F.cosine_similarity(anchor_embeddings, positive_embeddings)
                cosine_distance_pos = 1 - cosine_similarity_pos
                self.all_cosine_distances_positive.extend(cosine_distance_pos.cpu().numpy())

                # Anchor-Negative distances
                negative_labels = batch['negative_labels']
                negative_inputs_ids_list = batch['negative_input_ids']
                negative_attention_masks_list = batch['negative_attention_mask']

                for i in range(anchor_embeddings.size(0)): 
                    min_dist = dist_ax[i]
                    pred_label = anchor_labels[i] 
                    negative_input_ids = negative_inputs_ids_list[i, 0].to(self.device)
                    negative_attention_mask = negative_attention_masks_list[i, 0].to(self.device)
                    negative_embeddings = self.model(negative_input_ids.unsqueeze(0), negative_attention_mask.unsqueeze(0))
                    min_negative_embeddings = negative_embeddings

                    for neg_idx in range(negative_inputs_ids_list.size(1)):  # N_negatives per anchor = #num_labels - 1
                        negative_input_ids = negative_inputs_ids_list[i, neg_idx].to(self.device)
                        negative_attention_mask = negative_attention_masks_list[i, neg_idx].to(self.device)
                        negative_embeddings = self.model(negative_input_ids.unsqueeze(0), negative_attention_mask.unsqueeze(0))
                        dist_bx = self.distance_function(anchor_embeddings[i].unsqueeze(0), negative_embeddings)
                        if dist_bx < min_dist:
                            min_dist = dist_bx
                            pred_label = negative_labels[i][neg_idx]
                            min_negative_embeddings = negative_embeddings
                            print(f"+: {dist_ax[i]}, _: {dist_bx}, anchor: {anchor_labels[i]}, negative: {negative_labels[i][neg_idx]}")
                            # print(f"Min dist: {min_dist}, possible label: {correct_label}")
                    cosine_similarity_neg = F.cosine_similarity(anchor_embeddings[i], min_negative_embeddings)
                    cosine_distance_neg = 1 - cosine_similarity_neg
                    self.all_cosine_distances_negative.extend(cosine_distance_neg.cpu().numpy())

                    if pred_label == anchor_labels[i]:
                        correct += 1
                    total += 1

        accuracy = correct / total
        return accuracy

    def test_classification(self, test_dataloader):
        """
        Evaluates the classification model on the provided test data.
        Args:
            test_dataloader (DataLoader): DataLoader containing the test dataset.
        Returns:
            dict: A dictionary containing the following evaluation metrics:
                - 'accuracy' (float): The accuracy of the model on the test dataset.
                - 'precision' (float): The weighted precision of the model on the test dataset.
                - 'recall' (float): The weighted recall of the model on the test dataset.
                - 'f1_score' (float): The weighted F1 score of the model on the test dataset.
                - 'confusion_matrix' (ndarray): The confusion matrix of the model's predictions.
                - 'auc' (float): The area under the ROC curve (AUC) of the model on the test dataset.
        """
        if self.classification_model is None:
            print("Classification model not provided for testing.")
            return None
        self.classification_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch['anchor_input_ids'].to(self.device)
                attention_mask = batch['anchor_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.classification_model(input_ids, attention_mask)
                # probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
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

        classes = np.unique(all_labels)
        y_true_binarized = label_binarize(all_labels, classes=classes)
        y_pred_binarized = label_binarize(all_preds, classes=classes)
        auc = roc_auc_score(y_true_binarized, y_pred_binarized,labels = classes, average="macro", multi_class="ovo")

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'auc': auc
        }

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Macro AUC Score: {auc:.4f}")

        return results

    def plot_tsne_for_authors(self, dataloader):
        """
        Plots a t-SNE visualization for author embeddings.
        This function takes a dataloader containing batches of data, extracts embeddings
        using the model, and then applies t-SNE to reduce the dimensionality of these embeddings.
        The resulting 2D t-SNE plot is displayed and saved as an image file.
        Args:
            dataloader (DataLoader): A PyTorch DataLoader containing batches of data. Each batch
                                     should be a dictionary with keys 'anchor_input_ids', 
                                     'anchor_attention_mask', and 'label'.
        Returns:
            None
        Note:
            - The function assumes that the model and device are already set up as attributes of the class.
            - The function also assumes that `self.author_id_map` is a dictionary mapping author IDs to author names.
            - The t-SNE plot is saved in the directory specified by `self.repository_id`.
        """
        
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
        tsne = TSNE(n_components=num_classes, perplexity=80, max_iter=3000, method='exact')
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

    def plot_cosine_distence_distribution(self):
        """
        Plots the distribution of cosine distances between anchor/positive examples and anchor/negative examples.
        Args:
            test_dataloader (DataLoader): A PyTorch DataLoader containing batches of test data. Each batch
                                           should be a dictionary with keys 'anchor_input_ids', 'anchor_attention_mask',
                                           'positive_input_ids', 'positive_attention_mask', and 'label'.
        Returns:
            None
        Note:
            - The function assumes that the model and device are already set up as attributes of the class.
        """
        data = pd.DataFrame({
            "Cosine Distance": np.concatenate([self.all_cosine_distances_positive, self.all_cosine_distances_negative]),
            "Pair Type": ["Anchor-Positive"] * len(self.all_cosine_distances_positive) + ["Anchor-Negative"] * len(self.all_cosine_distances_negative)
        })

        plt.figure(figsize=(6, 8))
        sns.violinplot(data=data, x="Pair Type", y="Cosine Distance", split=True, hue="Pair Type", inner="quartile")
        plt.ylabel("Cosine Distance")
        plt.title("Distribution of Cosine Distances Between Embeddings")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.repository_id}/cosine_distances.png")
        



