import copy
from typing import Any, Tuple
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler


def train_one_epoch(
      model: nn.Module,
      dataloader: DataLoader,
      optimizer: Optimizer,
      scheduler: Scheduler,
      loss_function: nn.Module,
      device: str) -> Tuple[float, float]:
  model.train()
  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits, text_loss = model(images)
    loss = loss_function(logits, labels) + text_loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    total_loss += loss.item() * images.size(0)
    total_correct += (logits.argmax(dim=1) == labels).sum().item()
    total_samples += images.size(0)

  avg_loss = total_loss / total_samples
  accuracy = total_correct / total_samples
  return avg_loss, accuracy


def evaluate(
      model: nn.Module,
      dataloader: DataLoader,
      device: str
      ) -> float:
  model.eval()
  total_correct = 0
  total_samples = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images = images.to(device)
      labels = labels.to(device)

      logits, _ = model(images)
      total_correct += (logits.argmax(dim=1) == labels).sum().item()
      total_samples += images.size(0)

  accuracy = total_correct / total_samples
  return accuracy


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: Optimizer,
                scheduler: Scheduler,
                loss_function: nn.Module,
                num_epochs: int,
                device: str
                ) -> Tuple[
                    float,
                    dict,
                    dict
                ]:
    # Training loop
    print("Starting training...")
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_val_acc = 0
    best_model = nn.Identity().state_dict()

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_function, device)

        # Validation phase
        val_acc = evaluate(model, val_loader, device)

        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    return best_val_acc, best_model, training_history


def config_to_key(config: dict[str, dict[str, Any]]) -> tuple:
    """Convert config dict to a hashable tuple for caching"""
    key_config = config.copy()
    key_config['tunable'] = key_config['tunable'].copy()  # shallow copy of sub-dict
    key_config['tunable'].pop('clip_model', None)
    key_config['tunable'].pop('all_classnames', None)
    key_config['tunable'].pop('active_classnames', None)
    return tuple(sorted(key_config['tunable'].items()))

def build_context_and_evaluate_config(
        train_dataset,
        dev_dataset,
        config: dict,
        loss_function: nn.Module,
        model_class: type,
        optimizer_class: type,
        scheduler_class: type
):
    # Build model, optimizer, and scheduler
    model = model_class(**config['tunable']).to(config['fixed']['device'])
    optimizer = optimizer_class(model.parameters(), lr=config['tunable']['learning_rate'], momentum=config['tunable']['momentum'])
    scheduler = scheduler_class(optimizer, T_max=config['tunable']['T_max'], eta_min=config['tunable']['eta_min'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['tunable']['train_batch_size'], shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=config['tunable']['val_batch_size'])

    best_val_acc, best_model, training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        num_epochs=config['tunable']['num_epochs'],
        device=config['fixed']['device']
    )
    model.load_state_dict(best_model)

    return model, best_val_acc, training_history

def hill_climbing_hyperparameter_search(
        train_dataset,
        val_dataset,
        initial_config: dict,
        hyperparams: dict[str, dict[str, list]],
        loss_function: nn.Module,
        model_class: type,
        optimizer_class: type,
        scheduler_class: type,
        max_steps: int = 5
        ) -> tuple[torch.nn.Module, dict, float, list[dict]]:
    config_cache = {}
    current_config = initial_config
    steps = []
    i=0
    first_step_config = initial_config.copy()
    first_step_config['tunable'] = first_step_config['tunable'].copy()
    log_matrix = {
        'step': 0,
        'initial_step_config': first_step_config,
        'explored_configs': []
    }
    steps.append(log_matrix)

    def get_or_evaluate_config(
            config: dict,
            train_dataset,
            val_dataset,
            loss_function,
            model_class,
            optimizer_class,
            scheduler_class,
            ) -> tuple[torch.nn.Module, float]:
        key = config_to_key(config)
        if key in config_cache:
            print(f"Using cached result: Val Acc= {config_cache[key]:.2f}")
            return torch.nn.Identity(), config_cache[key]
        else:
            model, acc, _ = build_context_and_evaluate_config( #TODO: decide if it's worth keeping also training history
                train_dataset=train_dataset,
                dev_dataset=val_dataset,
                config=config,
                loss_function=loss_function,
                model_class=model_class,
                optimizer_class=optimizer_class,
                scheduler_class=scheduler_class
            )
            config_cache[key] = acc
            print(f"Computed Val Acc = {acc:.2f} (cached)")
            return model, acc

    # Evaluate starting point
    current_model, current_acc = get_or_evaluate_config(
        current_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_function=loss_function,
        model_class=model_class,
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_class
    )

    while True:
        log_step = log_matrix.copy()
        log_step['step'] = i
        log_step['current_config'] = current_config.copy()
        log_step['current_config']['tunable'] = log_step['current_config']['tunable'].copy()
        log_step['current_config']['tunable'].pop('clip_model', None)
        log_step['current_acc'] = current_acc
        print(f"Current Acc: {current_acc:.2f}")

        best_neighbor_config = None
        best_neighbor_model = current_model # Assignment for pylance compliance
        best_neighbor_acc = current_acc  # Start with current Acc as baseline

        # Try changing each hyperparameter one at a time
        for param_name, param_values in hyperparams['tunable'].items():
            current_value = current_config['tunable'][param_name]
            current_idx = param_values.index(current_value)
            
            # Try adjacent values (neighbors)
            neighbors = []
            if current_idx > 0:  # Can go down
                neighbors.append(param_values[current_idx - 1])
            if current_idx < len(param_values) - 1:  # Can go up
                neighbors.append(param_values[current_idx + 1])
            
            for neighbor_value in neighbors:
                # Create neighbor configuration
                neighbor_config = current_config.copy()
                neighbor_config['tunable'] = neighbor_config['tunable'].copy()
                neighbor_config['tunable'][param_name] = neighbor_value
                
                print(f"\nTrying {param_name}: {current_value} -> {neighbor_value}")
                model, neighbor_acc = get_or_evaluate_config(
                    config=neighbor_config,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    loss_function=loss_function,
                    model_class=model_class,
                    optimizer_class=optimizer_class,
                    scheduler_class=scheduler_class
                )
                neighbor_config_copy = neighbor_config.copy()
                neighbor_config_copy['tunable'] = neighbor_config_copy['tunable'].copy()
                neighbor_config_copy.pop('clip_model', None)
                log_step['explored_configs'].append({
                    'config': neighbor_config_copy,
                    'acc': neighbor_acc
                })

                # Keep track of best neighbor
                if neighbor_acc > best_neighbor_acc:
                    best_neighbor_acc = neighbor_acc
                    best_neighbor_config = neighbor_config.copy()
                    best_neighbor_config['tunable'] = best_neighbor_config['tunable'].copy()
                    best_neighbor_model = model

        # Check if we found improvement
        if best_neighbor_config and best_neighbor_acc > current_acc and i < max_steps:
            print(f"Found better value from {current_acc} to {best_neighbor_acc}")
            current_config = best_neighbor_config
            current_acc = best_neighbor_acc
            current_model = best_neighbor_model
        else:
            print(f"\nNo improvement found or maximum iterations reached. Stopping hill climbing.")
            final_config = current_config.copy()
            final_config['tunable'] = final_config['tunable'].copy()
            final_config.pop('clip_model', None)
            log_step['final_step_config'] = final_config
            log_step['final_step_acc'] = current_acc
            steps.append(log_step)
            break
        steps.append(log_step)  # Log the step
        i += 1

    print(f"\n=== Final Results ===")
    print(f"Total configurations evaluated: {len(config_cache)}")
    print(f"Best configuration: {final_config}")
    print(f"Best Acc: {current_acc:.2f}")

    return current_model, current_config, current_acc, steps