import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def calculate_confusion(output: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix.

    Args:
        output: Output of model.
        label: Ground truth label.
    """
    classNum = len(np.unique(label))
    confusion = np.zeros((classNum, classNum), dtype=np.uint32)
    output = output.argmax(axis=1)
    for ground_truth in range(classNum):
        for predict in range(classNum):
            confusion[ground_truth][predict] = (
                output[label == ground_truth] == predict
            ).sum()
    return confusion


class EvalRecord:
    """Class for recording evaluation result.

    Attributes:
        label: :class:`numpy.ndarray` of shape (n,).
            Ground truth label.
        output: :class:`numpy.ndarray` of shape (n, classNum).
            Output of model.
        gradient: dict of :class:`numpy.ndarray` of shape (n, classNum, ...) with
                  class index as key.
            Gradient of model by class index.
    """
    def __init__(self, label: np.ndarray, output: np.ndarray, gradient: dict, gradient_input:dict, smoothgrad: dict, smoothgrad_sq: dict, vargrad:dict) -> None:
        self.label = label
        self.output = output
        self.gradient = gradient
        self.gradient_input = gradient_input
        self.smoothgrad = smoothgrad
        self.smoothgrad_sq = smoothgrad_sq
        self.vargrad = vargrad

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
        index_header_str = ",".join([str(i) for i in range(self.output.shape[1])])
        header = f'{index_header_str},ground_truth,predict'
        np.savetxt(
            target_path, data, delimiter=',', newline='\n',
            header=header, comments='')
        

    def export_saliency(self, method:str, target_path: str) -> None:
        """ Export saliency map as torch file.
        Args:
            method: saliency type to be exported.
            target_path: Path to save saliency map.
        """
        if method == "Gradient":
            saliency = self.gradient
        elif method == "Gradient * Input":
            saliency = self.gradient_input
        elif method == "SmoothGrad":
            saliency = self.smoothgrad
        elif method == "SmoothGrad_Squared":
            saliency = self.smoothgrad_sq
        elif method == "VarGrad":
            saliency = self.vargrad
        return saliency
    #
    def get_acc(self) -> float:
        """Get accuracy of the model."""
        return sum(self.output.argmax(axis=1) == self.label) / len(self.label)

    def get_auc(self) -> float:
        """Get auc of the model."""
        if torch.nn.functional.softmax(
            torch.Tensor(self.output), dim=1
        ).numpy().shape[-1] <=2:
            return roc_auc_score(
                self.label,
                torch.nn.functional.softmax(
                    torch.Tensor(self.output), dim=1
                ).numpy()[:, -1]
            )
        else:
            return roc_auc_score(
                self.label,
                torch.nn.functional.softmax(
                    torch.Tensor(self.output), dim=1
                ).numpy(),
                multi_class='ovr'
            )

    def get_kappa(self) -> float:
        """Get kappa of the model."""
        confusion = calculate_confusion(self.output, self.label)
        classNum = len(confusion)
        P0 = np.diagonal(confusion).sum() / confusion.sum()
        Pe = (
            sum([confusion[:, i].sum() * confusion[i].sum() for i in range(classNum)]) /
            (confusion.sum() * confusion.sum())
        )
        return (P0 - Pe) / (1 - Pe)
    
    def get_gradient(self, labelIndex: int) -> np.ndarray:
        """Return gradient of model by class index."""
        return self.gradient[labelIndex]
    
    def get_gradient_input(self, labelIndex: int) -> np.ndarray:
        """Return gradient times input of model by class index."""
        return self.gradient_input[labelIndex]

    def get_smoothgrad(self, labelIndex: int) -> np.ndarray:
        """Return smoothgrad of model by class index."""
        return self.smoothgrad[labelIndex]
    
    def get_smoothgrad_sq(self, labelIndex: int) -> np.ndarray: 
        """Return smoothgrad squared of model by class index."""
        return self.smoothgrad_sq[labelIndex]
    
    def get_vargrad(self, labelIndex: int) -> np.ndarray:
        """Return vargrad of model by class index."""
        return self.vargrad[labelIndex]