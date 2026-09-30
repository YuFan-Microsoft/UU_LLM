from deepspeed_llm_trainer import main


USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": ["User_Profile_L1_gpt54", "User_Profile_L2_gpt54"],
    "dataset_shuffle_seed": 42,
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS)