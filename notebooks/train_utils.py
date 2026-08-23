import torch
from torch import nn
import tqdm


def cls_train_one_epoch(
    model, train_dl, criterion, optimizer, metrics_calculator, device="cuda"
):
    """
    Performs the optimization over the whole dataset once.
    Resets the metrics_calculator at the beggining and updates metrics inplace but does not return the metrics directly.
    Those can be accesed outside of this function.

    Args:
            model: Model trained.
            train_dl: Dataloader with data used for training.
            criterion: Criterion used for calculating the loss.
            optimizer: Optimizer used for updating model's weights.
            metrics_calculator: Instance of metrics calculator used for computing metrics on train dataset.
            device (optional): Device used for calculations like CUDA or CPU.
    Returns:
            float: Mean training loss for current epoch.
    """

    metrics_calculator.reset()
    model.train()
    train_epoch_loss = 0
    n_instances = 0

    for inputs, targets in tqdm.tqdm(train_dl, desc="Training"):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        metrics_calculator.update(outputs.detach(), targets.detach())
        n_instances += inputs.shape[0]

    mean_epoch_loss = train_epoch_loss / n_instances
    return mean_epoch_loss


@torch.no_grad()
def cls_evaluate(model, dataloader, criterion, metrics_calculator, device="cuda"):
    """
    Evaluates the model given a dataloader

    Args:
            model: Model to evaluate.
            dataloader: Dataloader with data used for training.
            criterion: Criterion used for calculating the loss.
            metrics_calculator: Instance of metrics calculator used for computing metrics.
            device (optional): Device used for calculations like CUDA or CPU.
    Returns:
            float: Mean loss.
    """

    metrics_calculator.reset()
    model.eval()
    total_loss = 0
    n_instances = 0

    for inputs, targets in tqdm.tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        total_loss += loss.item()
        metrics_calculator.update(outputs.detach(), targets.detach())
        n_instances += inputs.shape[0]

    mean_loss = total_loss / n_instances
    return mean_loss


def cls_train(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    metrics_calculator,
    epochs=10,
    device="cuda",
):
    """
    Trains the model for the specified number of epochs.

    Args:
            model: Model trained.
            train_dl: Dataloader with training data.
            val_dl: Dataloader with validation data.
            criterion: Criterion used for calculating the loss.
            optimizer: Optimizer used to update model parameters.
            metrics_calculator: Instance of metrics calculator used for computing metrics.
            epochs (optional): Number of epochs to train the model for.
            device (optional): Device used for calculations like CUDA or CPU.
    Returns:
            list[float]: List of training loss in each epoch.
            list[float]: List of validation loss in each epoch.
    """
    best_weights = model.state_dict()
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = cls_train_one_epoch(
            model, train_dl, criterion, optimizer, metrics_calculator, device
        )
        accuracy, precision, recall, f1_score, auprc, auroc = (
            metrics_calculator.compute_all()
        )
        val_loss = cls_evaluate(model, val_dl, criterion, metrics_calculator, device)
        accuracy, precision, recall, f1_score, auprc, auroc = (
            metrics_calculator.compute_all()
        )

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, AUPRC: {auprc:.4f}, AUROC: {auroc:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    return train_losses, val_losses


def sae_train_one_epoch(model, train_dl, criterion, optimizer, device="cuda"):
    """
    Performs the SAE optimization over the whole dataset once.

    Args:
            model: Model trained.
            train_dl: Dataloader with data used for training.
            criterion: Criterion used for calculating the loss.
            optimizer: Optimizer used for updating model's weights.
            device (optional): Device used for calculations like CUDA or CPU.
    Returns:
            float: Mean training loss for current epoch.
            float: Mean number of active hidden vector neurons in each batch.
    """

    model.train()
    train_epoch_loss = 0
    total_active_neurons = 0
    n_instances = 0

    is_topk = isinstance(criterion, nn.MSELoss)

    for sae_inputs in tqdm.tqdm(train_dl, desc="Training"):
        optimizer.zero_grad()

        # data.TensorDataset returns list of len 1 so this retrieves all batches as torch.Tensor
        sae_inputs = sae_inputs[0].to(device)

        outputs, hidden = model(sae_inputs)

        if is_topk:
            loss = criterion(outputs, sae_inputs)
        else:
            loss = criterion(outputs, sae_inputs, hidden)

        active_neurons = torch.sum(hidden > 0)

        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        total_active_neurons += active_neurons.item()
        n_instances += sae_inputs.shape[0]

    mean_epoch_loss = train_epoch_loss / n_instances
    mean_active_neurons = total_active_neurons / n_instances
    return mean_epoch_loss, mean_active_neurons


@torch.no_grad()
def sae_evaluate(model, dataloader, criterion, device="cuda"):
    """
    Evaluates SAE given a dataloader

    Args:
            model: Model to evaluate.
            dataloader: Dataloader with data used for training.
            criterion: Criterion used for calculating the loss.
            device (optional): Device used for calculations like CUDA or CPU.
    Returns:
            float: Mean loss.
            float: Mean number of active hidden vector neurons in each batch.
    """

    model.eval()
    total_loss = 0
    total_active_neurons = 0
    n_instances = 0

    is_topk = isinstance(criterion, nn.MSELoss)

    for sae_inputs in tqdm.tqdm(dataloader, desc="Evaluating"):
        # data.TensorDataset returns list of len 1 so this retrieves all batches as torch.Tensor
        sae_inputs = sae_inputs[0].to(device)

        outputs, hidden = model(sae_inputs)

        if is_topk:
            loss = criterion(outputs, sae_inputs)
        else:
            loss = criterion(outputs, sae_inputs, hidden)

        active_neurons = torch.sum(hidden > 0)

        total_loss += loss.item()
        total_active_neurons += active_neurons.item()
        n_instances += sae_inputs.shape[0]

    mean_loss = total_loss / n_instances
    mean_active_neurons = total_active_neurons / n_instances
    return mean_loss, mean_active_neurons


def sae_train(model, train_dl, val_dl, criterion, optimizer, epochs=10, device="cuda"):
    """
    Function to train SAE model using dataset with embeddings from image classification model.

    Args:
            model: SAE model to train.
            train_dl: Dataset with training embeddings.
            val_dl: Dataset with validation embeddings.
            criterion: Criterion used to calculate SAE loss.
            optimizer: Optimizer used to update SAE params.
            epochs (optional): Number of epochs to train SAE for.
            device (optional): Device used for calculations like CUDA or CPU.
    """

    best_weights = model.state_dict()
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss, train_mean_active_neurons = sae_train_one_epoch(
            model, train_dl, criterion, optimizer, device
        )
        val_loss, val_mean_active_neurons = sae_evaluate(
            model, val_dl, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train active neurons: {train_mean_active_neurons:.4f}, Val active neurons: {val_mean_active_neurons:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)

    return train_losses, val_losses
