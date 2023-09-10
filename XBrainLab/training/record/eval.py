import os
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_confusion(output: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix.
    
    Args:
        output: Output of model.
        label: Ground truth label.
    """
    classNum = len(np.unique(label))
    confusion = np.zeros((classNum,classNum), dtype=np.uint32)
    output = output.argmax(axis=1)
    for ground_truth in range(classNum):
        for predict in range(classNum):
            confusion[ground_truth][predict] = (output[label == ground_truth] == predict).sum()
    return confusion


class EvalRecord:
    """Class for recording evaluation result.

    Attributes:
        label: :class:`numpy.ndarray` of shape (n,).
            Ground truth label.
        output: :class:`numpy.ndarray` of shape (n, classNum).
            Output of model.
        gradient: dict of :class:`numpy.ndarray` of shape (n, classNum, ...) with class index as key.
            Gradient of model by class index.
    """
    def __init__(self, label: np.ndarray, output: np.ndarray, gradient: dict):
        self.label = label
        self.output = output
        self.gradient = gradient
    
    def export(self, target_path: str) -> None:
        """Export evaluation result as torch file.

        Args:
            target_path: Path to save evaluation result.
        """
        record = {
            'label': self.label,
            'output': self.output,
            'gradient': self.gradient,
        }
        torch.save(record, os.path.join(target_path, 'eval'))
    
    def export_csv(self, target_path: str) -> None:
        """Export evaluation result as csv file.

        Args:
            target_path: Path to save evaluation result.
        """
        data = np.c_[self.output, self.label, self.output.argmax(axis=1)]
        np.savetxt(target_path, data, delimiter=',', newline='\n', header=f'{",".join([str(i) for i in range(self.output.shape[1])])},ground_truth,predict', comments='')
    #
    def get_acc(self) -> float:
        """Get accuracy of the model."""
        return sum(self.output.argmax(axis=1) == self.label) / len(self.label)

    def get_auc(self) -> float:
        """Get auc of the model."""
        if torch.nn.functional.softmax(torch.Tensor(self.output), dim=1).numpy().shape[-1] <=2:
            return roc_auc_score(self.label, torch.nn.functional.softmax(torch.Tensor(self.output), dim=1).numpy()[:,-1])
        else:
            return roc_auc_score(self.label, torch.nn.functional.softmax(torch.Tensor(self.output), dim=1).numpy(), multi_class='ovr')

    def get_kappa(self) -> float:
        """Get kappa of the model."""
        confusion = calculate_confusion(self.output, self.label)
        classNum = len(confusion)
        P0 = np.diagonal(confusion).sum() / confusion.sum()
        Pe = sum([confusion[:,i].sum() * confusion[i].sum() for i in range(classNum)]) / (confusion.sum() * confusion.sum())
        return (P0 - Pe) / (1 - Pe)
