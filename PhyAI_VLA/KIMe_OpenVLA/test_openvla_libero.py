# !pip install --quiet mediapy
# !pip install robosuite==1.4

from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor

from experiments.robot.libero.libero_utils import (
    get_libero_image,
    quat2axisangle,
)

from typing import Optional, Union
from pathlib import Path
import os 
import mediapy
import tqdm
import numpy as np
import cv2

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

# from tqdm.notebook import tqdm
from tqdm import tqdm


class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 0                                    # Random Seed (for reproducibility)

    resize_size: int = 512                          


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def create_folder_if_not_exists(file_path):
    # 파일 경로에서 폴더 경로 추출
    folder_path = os.path.dirname(file_path)
    
    # 폴더가 비어있지 않고, 존재하지 않으면 폴더 생성
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"폴더 생성: {folder_path}")
    else:
        print(f"폴더가 이미 존재하거나 경로가 비어있습니다: {folder_path}")


def save_rollout_video_imageio(rollout_images, idx):
    import imageio
    """Saves an MP4 replay of an episode."""
    mp4_path = f"./video/demo_task_{idx}.mp4"
    create_folder_if_not_exists(mp4_path)
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"[imageio] Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def save_rollout_video_moviepy(rollout_images, idx):
    from moviepy import ImageSequenceClip

    """Saves an MP4 replay of an episode."""
    mp4_path = f"./video/demo_task_{idx}.mp4"
    create_folder_if_not_exists(mp4_path)
    clip = ImageSequenceClip(rollout_images, fps=30)
    clip.write_videofile("output4.mp4", codec='libx264', audio=False)
    print(f"[moviepy] Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def init_test(cfg: GenerateConfig, task_id: int):
    # Set random seed
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

        # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

    # Get task
    task = task_suite.get_task(task_id)

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    env, task_description = get_libero_env(task, resolution=256)

    print('task_description:')
    print(task_description)

    print(f"\nTask: {task_description}")


    return {"cfg": cfg, "initial_states": initial_states, "model": model, "processor": processor, "env": env, "task": task, "instruction": task_description}


def test(env_info, task_successes, episode_idx):
    env = env_info["env"]
    initial_states = env_info["initial_states"]
    cfg = env_info["cfg"]
    task = env_info["task"]
    model = env_info["model"]
    processor = env_info["processor"]
    instruction = env_info["instruction"]


    # Reset environment
    env.reset()

    # Set initial states
    obs = env.set_init_state(initial_states[episode_idx])

    # Setup
    t = 0
    replay_images = []
    if cfg.task_suite_name == "libero_spatial":
        max_steps = 400*2  # longest training demo has 193 steps
    elif cfg.task_suite_name == "libero_object":
        max_steps = 400*2  # longest training demo has 254 steps
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 400*2  # longest training demo has 270 steps
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520*2  # longest training demo has 505 steps
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400*2  # longest training demo has 373 steps

    print(f"Starting task : {task}...")

    sim_steps = list(range(max_steps + cfg.num_steps_wait))
    for t in tqdm(sim_steps, desc="처리 중"):
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                get_libero_dummy_action = [0, 0, 0, 0, 0, 0, -1]
                obs, reward, done, info = env.step(get_libero_dummy_action)
                continue

            # Get preprocessed image
            img = get_libero_image(obs, cfg.resize_size)

            # Save preprocessed image for replay video
            replay_images.append(img)            
            # cv2.imshow("Real-time View", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Prepare observations dict
            # Note: OpenVLA does not take proprio state as input
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }

            # Query model to get action
            action = get_action(
                cfg,
                model,
                observation,
                instruction,
                processor=processor,
            )

            # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
            action = normalize_gripper_action(action, binarize=True)

            # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
            # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                task_successes += 1
                print('Success', task_successes)
                break

        except Exception as e:
            print(f"Caught exception: {e}")
            break
    
    return replay_images

 

def main():

    cfg = GenerateConfig()
    # ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    cfg.task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-libero-10"
    cfg.load_in_4bit = True
    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    cfg.resize_size = 512
    
    task_id = 6
    env_info = init_test(cfg, task_id)


    task_successes = 0
    episode_idx = 0
    # env_info["instruction"] = instruction

    replay_images = test(env_info, task_successes, episode_idx)

    env_info["env"].close()

    save_rollout_video_imageio(replay_images, 0)
    save_rollout_video_moviepy(replay_images, 1)

if __name__ == "__main__":
    main()
