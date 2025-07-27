import torch
import warnings
from gr00t.utils.eval import calc_mse_for_single_trajectory
from gr00t.model.policy import Gr00tPolicy

# %%
# Set up paths and device
dataset_path = "./datasets/G1_BlockStacking_Dataset"
finetuned_model_path = "./models"  # Update to your actual checkpoint path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Re-create the same modality configs and transforms as training
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.model.transforms import GR00TTransform

embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

# Modality configs (same as training)
video_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["video.cam_right_high"],
)

state_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
)

action_modality = ModalityConfig(
    delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    modality_keys=["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
)

language_modality = ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.task_description"],
)

modality_config = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}

# Transforms (same as training)
modality_transform = ComposedModalityTransform(
    transforms=[
        # video transforms
        VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
        VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
        VideoResize(apply_to=video_modality.modality_keys, height=224, width=224, interpolation="linear", backend="torchvision"),
        VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
        VideoToNumpy(apply_to=video_modality.modality_keys),

        # state transforms
        StateActionToTensor(apply_to=state_modality.modality_keys),
        StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
            "state.left_arm": "min_max",
            "state.right_arm": "min_max",
            "state.left_hand": "min_max",
            "state.right_hand": "min_max",
        }),

        # action transforms
        StateActionToTensor(apply_to=action_modality.modality_keys),
        StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
            "action.right_arm": "min_max",
            "action.left_arm": "min_max",
            "action.right_hand": "min_max",
            "action.left_hand": "min_max",
        }),

        # ConcatTransform
        ConcatTransform(
            video_concat_order=video_modality.modality_keys,
            state_concat_order=state_modality.modality_keys,
            action_concat_order=action_modality.modality_keys,
        ),
        # model-specific transform
        GR00TTransform(
            state_horizon=len(state_modality.delta_indices),
            action_horizon=len(action_modality.delta_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
)

# %%
# Load evaluation dataset (without transforms for the eval function)
from gr00t.data.dataset import LeRobotSingleDataset

dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_config,
    embodiment_tag=embodiment_tag,
    video_backend="torchvision_av",
    # Note: don't apply transforms here, the eval function handles them
)

print(f"Dataset size: {len(dataset)}")

# %%
# Create the fine-tuned policy
finetuned_policy = Gr00tPolicy(
    model_path=finetuned_model_path,
    embodiment_tag="new_embodiment",
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

print("Fine-tuned policy loaded successfully!")

# %%
# Suppress warnings for cleaner output
warnings.simplefilter("ignore", category=FutureWarning)

# %%
# Evaluate specific trajectories
def evaluate_multiple_trajectories(policy, dataset, traj_ids, modality_keys, steps=150, action_horizon=16):
    """
    Evaluate multiple trajectories and return MSE results
    """
    results = {}

    for traj_id in traj_ids:
        print(f"\n=== Evaluating Trajectory {traj_id} ===")
        try:
            mse = calc_mse_for_single_trajectory(
                policy,
                dataset,
                traj_id=traj_id,
                modality_keys=modality_keys,
                steps=steps,
                action_horizon=action_horizon,
                plot=True
            )
            results[traj_id] = mse
            print(f"MSE loss for trajectory {traj_id}: {mse}")
        except Exception as e:
            print(f"Error evaluating trajectory {traj_id}: {e}")
            results[traj_id] = None

    return results

# %%
# Evaluate all 4 action modalities separately
print("=" * 60)
print("EVALUATING ALL ACTION MODALITIES")
print("=" * 60)

# Test trajectory 0 with all modalities
all_modalities_mse = calc_mse_for_single_trajectory(
    finetuned_policy,
    dataset,
    traj_id=0,
    modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    steps=150,
    action_horizon=16,
    plot=True
)
print(f"MSE loss for trajectory 0 (all modalities): {all_modalities_mse}")

# %%
# Evaluate arms only
print("\n" + "=" * 60)
print("EVALUATING ARMS ONLY")
print("=" * 60)

arms_mse = calc_mse_for_single_trajectory(
    finetuned_policy,
    dataset,
    traj_id=0,
    modality_keys=["left_arm", "right_arm"],
    steps=150,
    action_horizon=16,
    plot=True
)
print(f"MSE loss for trajectory 0 (arms only): {arms_mse}")

# %%
# Evaluate hands only
print("\n" + "=" * 60)
print("EVALUATING HANDS ONLY")
print("=" * 60)

hands_mse = calc_mse_for_single_trajectory(
    finetuned_policy,
    dataset,
    traj_id=0,
    modality_keys=["left_hand", "right_hand"],
    steps=150,
    action_horizon=16,
    plot=True
)
print(f"MSE loss for trajectory 0 (hands only): {hands_mse}")

# %%
# Evaluate right side only (as in your example)
print("\n" + "=" * 60)
print("EVALUATING RIGHT SIDE ONLY")
print("=" * 60)

right_side_mse = calc_mse_for_single_trajectory(
    finetuned_policy,
    dataset,
    traj_id=0,
    modality_keys=["right_arm", "right_hand"],
    steps=150,
    action_horizon=16,
    plot=True
)
print(f"MSE loss for trajectory 0 (right side): {right_side_mse}")

# %%
# Evaluate multiple trajectories for statistical significance
print("\n" + "=" * 60)
print("EVALUATING MULTIPLE TRAJECTORIES")
print("=" * 60)

# Test on first 5 trajectories
trajectory_ids = list(range(min(5, len(dataset))))
multi_traj_results = evaluate_multiple_trajectories(
    finetuned_policy,
    dataset,
    traj_ids=trajectory_ids,
    modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    steps=100,  # Shorter for faster evaluation
    action_horizon=16
)

# %%
# Calculate summary statistics
valid_mses = [mse for mse in multi_traj_results.values() if mse is not None]

if valid_mses:
    import numpy as np
    mean_mse = np.mean(valid_mses)
    std_mse = np.std(valid_mses)
    min_mse = np.min(valid_mses)
    max_mse = np.max(valid_mses)

    print(f"\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Number of evaluated trajectories: {len(valid_mses)}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Standard deviation: {std_mse:.6f}")
    print(f"Min MSE: {min_mse:.6f}")
    print(f"Max MSE: {max_mse:.6f}")

    # Save results
    results_summary = {
        "model_checkpoint": finetuned_model_path,
        "dataset_path": dataset_path,
        "all_modalities_mse": float(all_modalities_mse),
        "arms_only_mse": float(arms_mse),
        "hands_only_mse": float(hands_mse),
        "right_side_mse": float(right_side_mse),
        "multi_trajectory_results": {str(k): float(v) if v is not None else None for k, v in multi_traj_results.items()},
        "summary_stats": {
            "mean_mse": float(mean_mse),
            "std_mse": float(std_mse),
            "min_mse": float(min_mse),
            "max_mse": float(max_mse),
            "num_trajectories": len(valid_mses)
        }
    }

    import json
    results_file = f"{finetuned_model_path}/detailed_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetailed evaluation results saved to: {results_file}")
else:
    print("No valid MSE results obtained!")

print("\nEvaluation complete!")
