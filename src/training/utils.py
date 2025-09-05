"""
Training Utilities for Hybrid GNN-RNN Framework

This module provides utilities for training the hybrid model, including
loss functions, metrics, and training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import wandb
import time
from tqdm import tqdm


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with automatic loss weighting.
    """
    
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        uncertainty_weighting: bool = True,
        adaptive_weighting: bool = True
    ):
        super(MultiTaskLoss, self).__init__()
        
        self.task_weights = task_weights or {}
        self.uncertainty_weighting = uncertainty_weighting
        self.adaptive_weighting = adaptive_weighting
        
        # Initialize learnable weights for adaptive weighting
        if adaptive_weighting:
            self.log_vars = nn.ParameterDict()
    
    def add_task(self, task_name: str, loss_type: str = "mse"):
        """Add a new task to the multi-task loss."""
        if self.adaptive_weighting:
            self.log_vars[task_name] = nn.Parameter(torch.zeros(1))
        
        if task_name not in self.task_weights:
            self.task_weights[task_name] = 1.0
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        uncertainties: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions
            targets (Dict[str, torch.Tensor]): Ground truth targets
            uncertainties (Dict[str, torch.Tensor], optional): Uncertainty estimates
            
        Returns:
            Dict[str, torch.Tensor]: Loss components and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for task_name in predictions:
            if task_name not in targets:
                continue
            
            pred = predictions[task_name]
            target = targets[task_name]
            
            # Compute base loss
            if "efficiency" in task_name:
                # Regression loss for efficiency
                base_loss = F.mse_loss(pred, target)
            elif "maturation" in task_name and "logits" in task_name:
                # Classification loss for maturation
                base_loss = F.cross_entropy(pred, target.long())
            elif "maturation" in task_name and "probs" in task_name:
                # Probability matching loss
                base_loss = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    target,
                    reduction='batchmean'
                )
            else:
                # Default to MSE
                base_loss = F.mse_loss(pred, target)
            
            # Apply uncertainty weighting if available
            if (self.uncertainty_weighting and 
                uncertainties and 
                task_name in uncertainties):
                uncertainty = uncertainties[task_name]
                # Uncertainty-weighted loss
                weighted_loss = (
                    0.5 * torch.exp(-uncertainty) * base_loss + 
                    0.5 * uncertainty
                )
            else:
                weighted_loss = base_loss
            
            # Apply adaptive weighting
            if self.adaptive_weighting and task_name in self.log_vars:
                log_var = self.log_vars[task_name]
                precision = torch.exp(-log_var)
                final_loss = precision * weighted_loss + log_var
            else:
                # Apply manual weights
                weight = self.task_weights.get(task_name, 1.0)
                final_loss = weight * weighted_loss
            
            losses[f"{task_name}_loss"] = final_loss
            total_loss += final_loss
        
        losses["total_loss"] = total_loss
        return losses


class DifferentiationLoss(nn.Module):
    """
    Specialized loss for cardiomyocyte differentiation efficiency.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",  # "mse", "mae", "huber", "beta"
        beta: float = 0.1,  # For beta loss
        delta: float = 1.0  # For Huber loss
    ):
        super(DifferentiationLoss, self).__init__()
        
        self.loss_type = loss_type
        self.beta = beta
        self.delta = delta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute differentiation efficiency loss.
        
        Args:
            predictions (torch.Tensor): Predicted efficiency [batch_size]
            targets (torch.Tensor): Target efficiency [batch_size]
            weights (torch.Tensor, optional): Sample weights [batch_size]
            
        Returns:
            torch.Tensor: Computed loss
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(predictions, targets, reduction='none')
        elif self.loss_type == "mae":
            loss = F.l1_loss(predictions, targets, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(predictions, targets, delta=self.delta, reduction='none')
        elif self.loss_type == "beta":
            # Beta loss for efficiency prediction
            diff = torch.abs(predictions - targets)
            loss = torch.where(
                diff < self.beta,
                0.5 * diff ** 2 / self.beta,
                diff - 0.5 * self.beta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss to encourage diverse representations.
    """
    
    def __init__(self, lambda_ortho: float = 0.01):
        super(OrthogonalityLoss, self).__init__()
        self.lambda_ortho = lambda_ortho
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality loss.
        
        Args:
            embeddings (torch.Tensor): Embedding matrix [batch_size, dim]
            
        Returns:
            torch.Tensor: Orthogonality loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute gram matrix
        gram = torch.mm(embeddings.t(), embeddings)
        
        # Orthogonality loss (penalize off-diagonal elements)
        identity = torch.eye(gram.size(0), device=gram.device)
        ortho_loss = F.mse_loss(gram, identity)
        
        return self.lambda_ortho * ortho_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative representations.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
        loss_type: str = "infonce"  # "infonce", "triplet"
    ):
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.margin = margin
        self.loss_type = loss_type
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings [batch_size, dim]
            labels (torch.Tensor): Labels for contrastive learning [batch_size]
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        if self.loss_type == "infonce":
            return self._infonce_loss(embeddings, labels)
        elif self.loss_type == "triplet":
            return self._triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown contrastive loss type: {self.loss_type}")
    
    def _infonce_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE loss implementation."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs
        batch_size = embeddings.size(0)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()
        
        # Remove self-similarity
        mask.fill_diagonal_(0)
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        pos_sim = (exp_sim * mask).sum(dim=1)
        neg_sim = sum_exp_sim - torch.diag(exp_sim).unsqueeze(1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        
        return loss.mean()
    
    def _triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Triplet loss implementation."""
        # This is a simplified version - in practice, you'd want
        # more sophisticated triplet mining
        
        batch_size = embeddings.size(0)
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.t()).float()
        neg_mask = 1 - pos_mask
        
        # Remove self-similarity
        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)
        
        # Get positive and negative distances
        pos_distances = distances * pos_mask
        neg_distances = distances * neg_mask + pos_mask * 1e9  # Large value for positive pairs
        
        # Hard positive and negative mining
        hardest_positive = pos_distances.max(dim=1)[0]
        hardest_negative = neg_distances.min(dim=1)[0]
        
        # Triplet loss
        loss = F.relu(hardest_positive - hardest_negative + self.margin)
        
        return loss.mean()


class MetricsCalculator:
    """
    Calculate various metrics for model evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = {}
        self.targets = {}
        self.losses = {}
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor] = None
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions (Dict[str, torch.Tensor]): Model predictions
            targets (Dict[str, torch.Tensor]): Ground truth targets
            losses (Dict[str, torch.Tensor], optional): Loss values
        """
        for key, pred in predictions.items():
            if key not in self.predictions:
                self.predictions[key] = []
                self.targets[key] = []
            
            if key in targets:
                self.predictions[key].append(pred.detach().cpu())
                self.targets[key].append(targets[key].detach().cpu())
        
        if losses:
            for key, loss in losses.items():
                if key not in self.losses:
                    self.losses[key] = []
                self.losses[key].append(loss.detach().cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute accumulated metrics.
        
        Returns:
            Dict[str, float]: Computed metrics
        """
        metrics = {}
        
        # Concatenate all predictions and targets
        all_preds = {}
        all_targets = {}
        
        for key in self.predictions:
            if self.predictions[key] and self.targets[key]:
                all_preds[key] = torch.cat(self.predictions[key], dim=0).numpy()
                all_targets[key] = torch.cat(self.targets[key], dim=0).numpy()
        
        # Compute task-specific metrics
        for key in all_preds:
            pred = all_preds[key]
            target = all_targets[key]
            
            if "efficiency" in key:
                # Regression metrics
                metrics[f"{key}_mse"] = mean_squared_error(target, pred)
                metrics[f"{key}_mae"] = mean_absolute_error(target, pred)
                metrics[f"{key}_r2"] = r2_score(target, pred)
                
                # Correlation
                corr = np.corrcoef(target, pred)[0, 1]
                metrics[f"{key}_correlation"] = corr if not np.isnan(corr) else 0.0
                
            elif "maturation" in key and "probs" in key:
                # Classification metrics
                pred_classes = np.argmax(pred, axis=1)
                target_classes = target if target.ndim == 1 else np.argmax(target, axis=1)
                
                metrics[f"{key}_accuracy"] = accuracy_score(target_classes, pred_classes)
                metrics[f"{key}_precision"] = precision_score(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                metrics[f"{key}_recall"] = recall_score(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                metrics[f"{key}_f1"] = f1_score(
                    target_classes, pred_classes, average='weighted', zero_division=0
                )
                
                # AUC if binary or with proper handling for multiclass
                try:
                    if len(np.unique(target_classes)) == 2:
                        metrics[f"{key}_auc"] = roc_auc_score(target_classes, pred[:, 1])
                    else:
                        metrics[f"{key}_auc"] = roc_auc_score(
                            target_classes, pred, multi_class='ovr', average='weighted'
                        )
                except:
                    metrics[f"{key}_auc"] = 0.0
        
        # Compute average losses
        for key in self.losses:
            if self.losses[key]:
                avg_loss = torch.stack(self.losses[key]).mean().item()
                metrics[f"avg_{key}"] = avg_loss
        
        return metrics


class Trainer:
    """
    Main trainer class for the hybrid model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_wandb: bool = False,
        project_name: str = "hybrid-gnn-rnn",
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        
        # Initialize loss functions
        self.criterion = MultiTaskLoss(
            uncertainty_weighting=True,
            adaptive_weighting=True
        )
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Move model to device
        self.model.to(device)
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(model)
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        self.metrics_calc.reset()
        
        total_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                node_features=batch['node_features'],
                edge_index=batch['edge_index'],
                edge_attr=batch.get('edge_attr'),
                pos=batch.get('pos'),
                batch=batch.get('batch'),
                temporal_features=batch.get('temporal_features'),
                temporal_mask=batch.get('temporal_mask')
            )
            
            # Compute losses
            targets = {key: batch[key] for key in batch if 'target' in key}
            uncertainties = {key: outputs[key] for key in outputs if 'uncertainty' in key}
            
            losses = self.criterion(outputs, targets, uncertainties)
            loss = losses['total_loss'] / self.accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_val
                )
            
            # Optimizer step
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            self.metrics_calc.update(outputs, targets, losses)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.accumulate_grad_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Final optimizer step if needed
        if total_batches % self.accumulate_grad_batches != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        # Compute epoch metrics
        metrics = self.metrics_calc.compute_metrics()
        metrics['epoch'] = epoch
        metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f"train/{key}": value for key, value in metrics.items()})
        
        return metrics
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            epoch (int): Current epoch number
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        self.metrics_calc.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    node_features=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_attr=batch.get('edge_attr'),
                    pos=batch.get('pos'),
                    batch=batch.get('batch'),
                    temporal_features=batch.get('temporal_features'),
                    temporal_mask=batch.get('temporal_mask')
                )
                
                # Compute losses
                targets = {key: batch[key] for key in batch if 'target' in key}
                uncertainties = {key: outputs[key] for key in outputs if 'uncertainty' in key}
                
                losses = self.criterion(outputs, targets, uncertainties)
                
                # Update metrics
                self.metrics_calc.update(outputs, targets, losses)
        
        # Compute validation metrics
        metrics = self.metrics_calc.compute_metrics()
        metrics['epoch'] = epoch
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f"val/{key}": value for key, value in metrics.items()})
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


if __name__ == "__main__":
    # Example usage
    print("Training utilities module loaded successfully!")
    
    # Test loss functions
    batch_size = 16
    
    # Test MultiTaskLoss
    criterion = MultiTaskLoss(adaptive_weighting=True)
    criterion.add_task("efficiency", "mse")
    criterion.add_task("maturation", "classification")
    
    predictions = {
        "efficiency": torch.randn(batch_size, 1),
        "maturation": torch.randn(batch_size, 3)
    }
    
    targets = {
        "efficiency": torch.randn(batch_size, 1),
        "maturation": torch.randint(0, 3, (batch_size,))
    }
    
    losses = criterion(predictions, targets)
    print(f"Test losses: {list(losses.keys())}")
    
    # Test metrics calculator
    metrics_calc = MetricsCalculator()
    metrics_calc.update(predictions, targets, losses)
    metrics = metrics_calc.compute_metrics()
    print(f"Test metrics: {list(metrics.keys())}")
