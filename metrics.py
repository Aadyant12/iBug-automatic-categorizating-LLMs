import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from typing import List, Union, Dict
import numpy.typing as npt
import pandas as pd

class MultiClassMetrics:
    """
    A class to calculate and display various metrics for multi-class classification problems.
    Handles both numerical and non-numerical labels.
    """
    
    def __init__(self, y_true: npt.ArrayLike, y_pred: npt.ArrayLike):
        """
        Initialize with true labels and predictions.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels (can be strings or numbers)
        y_pred : array-like of shape (n_samples,)
            Predicted labels (can be strings or numbers)
        """
        # Initialize label encoder
        self.le = LabelEncoder()
        
        # Fit encoder on both true and predicted labels to ensure all classes are captured
        self.le.fit(np.concatenate([y_true, y_pred]))
        
        # Transform labels to numerical format
        self.y_true_encoded = self.le.transform(y_true)
        self.y_pred_encoded = self.le.transform(y_pred)
        
        # Store original labels for reference
        self.y_true_original = y_true
        self.y_pred_original = y_pred
        
        self.classes = self.le.classes_
        self.n_classes = len(self.classes)
        
        # Create class mapping for reference
        self.class_mapping = dict(zip(range(self.n_classes), self.classes))
    
    def accuracy(self) -> float:
        """Calculate accuracy score."""
        return accuracy_score(self.y_true_encoded, self.y_pred_encoded)
    
    def precision(self, average: str = 'weighted') -> Union[float, Dict[str, float]]:
        """
        Calculate precision score.
        
        Parameters:
        -----------
        average : str, optional (default='weighted')
            'micro', 'macro', 'weighted', None (returns per-class scores)
            
        Returns:
        --------
        float or dict
            If average is None, returns dict mapping class labels to scores
        """
        scores = precision_score(self.y_true_encoded, self.y_pred_encoded, 
                               average=average, zero_division=0)
        
        if average is None:
            return dict(zip(self.classes, scores))
        return scores
    
    def recall(self, average: str = 'weighted') -> Union[float, Dict[str, float]]:
        """
        Calculate recall score.
        
        Parameters:
        -----------
        average : str, optional (default='weighted')
            'micro', 'macro', 'weighted', None (returns per-class scores)
            
        Returns:
        --------
        float or dict
            If average is None, returns dict mapping class labels to scores
        """
        scores = recall_score(self.y_true_encoded, self.y_pred_encoded, 
                            average=average, zero_division=0)
        
        if average is None:
            return dict(zip(self.classes, scores))
        return scores
    
    def f1(self, average: str = 'weighted') -> Union[float, Dict[str, float]]:
        """
        Calculate F1 score.
        
        Parameters:
        -----------
        average : str, optional (default='weighted')
            'micro', 'macro', 'weighted', None (returns per-class scores)
            
        Returns:
        --------
        float or dict
            If average is None, returns dict mapping class labels to scores
        """
        scores = f1_score(self.y_true_encoded, self.y_pred_encoded, 
                         average=average, zero_division=0)
        
        if average is None:
            return dict(zip(self.classes, scores))
        return scores
    
    def confusion_matrix(self, as_dataframe: bool = True) -> Union[npt.ArrayLike, pd.DataFrame]:
        """
        Calculate confusion matrix.
        
        Parameters:
        -----------
        as_dataframe : bool, optional (default=True)
            If True, returns pandas DataFrame with labeled axes
            
        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            Confusion matrix
        """
        cm = confusion_matrix(self.y_true_encoded, self.y_pred_encoded)
        
        if as_dataframe:
            return pd.DataFrame(
                cm,
                index=pd.Index(self.classes, name='True'),
                columns=pd.Index(self.classes, name='Predicted')
            )
        return cm
    
    def kappa(self) -> float:
        """Calculate Cohen's Kappa score."""
        return cohen_kappa_score(self.y_true_encoded, self.y_pred_encoded)
    
    def class_report(self, as_dict: bool = False) -> Union[str, Dict]:
        """
        Generate detailed classification report.
        
        Parameters:
        -----------
        as_dict : bool, optional (default=False)
            If True, returns dictionary instead of formatted string
        """
        return classification_report(
            self.y_true_encoded, 
            self.y_pred_encoded,
            target_names=self.classes,
            output_dict=as_dict
        )
    
    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 for each class.
        
        Returns:
        --------
        dict
            Dictionary with metrics for each class
        """
        return {
            class_label: {
                'precision': self.precision(average=None)[class_label],
                'recall': self.recall(average=None)[class_label],
                'f1': self.f1(average=None)[class_label]
            }
            for class_label in self.classes
        }

    def summary(self) -> Dict[str, float]:
        """
        Get a summary of the main metrics.
        
        Returns:
        --------
        dict
            Dictionary containing main evaluation metrics
        """
        return {
            'accuracy': self.accuracy(),
            'macro_precision': self.precision(average='macro'),
            'macro_recall': self.recall(average='macro'),
            'macro_f1': self.f1(average='macro'),
            # 'weighted_f1': self.f1(average='weighted'),
            'kappa': self.kappa()
        }
    
def plot_confusion_matrix(conf_matrix_df, figsize=(12, 8), 
                         fmt='.0f', annot_size=12, 
                         cmap='Blues', normalize=False):
    
    # Create figure and axis with specified size
    plt.figure(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        conf_matrix = conf_matrix_df.div(conf_matrix_df.sum(axis=1), axis=0) * 100
        fmt = '.1f'  # Show one decimal for percentages
    else:
        conf_matrix = conf_matrix_df
    
    # Create heatmap
    sns.heatmap(conf_matrix, 
                annot=True,          # Show numbers in cells
                fmt=fmt,             # Number format
                cmap=cmap,           # Color scheme
                cbar=True,           # Show color bar
                # square=True,         # Make cells square
                annot_kws={'size': annot_size})    # Color of lines between cells
    
    # Customize the plot
    plt.title('Confusion Matrix', pad=20, size=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Rotate axis labels if needed (good for long class names)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt

# # Example usage
# if __name__ == "__main__":
#     # Sample data with string labels
#     y_true = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird']
#     y_pred = ['cat', 'bird', 'dog', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird']
    
#     # Calculate metrics
#     metrics = MultiClassMetrics(y_true, y_pred)
    
#     # Print summary
#     print("Summary Metrics:")
#     for metric, value in metrics.summary().items():
#         print(f"{metric}: {value:.3f}")
    
#     print("\nPer-class metrics:")
#     for class_label, class_metrics in metrics.per_class_metrics().items():
#         print(f"\n{class_label}:")
#         for metric_name, value in class_metrics.items():
#             print(f"  {metric_name}: {value:.3f}")
    
#     print("\nConfusion Matrix:")
#     print(metrics.confusion_matrix())
    
#     print("\nDetailed Classification Report:")
#     print(metrics.class_report())