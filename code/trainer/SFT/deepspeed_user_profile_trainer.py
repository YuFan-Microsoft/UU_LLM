from deepspeed_llm_trainer import main
import user_profile_rules


# V2 configs and their share of the rows of every epoch (%). L1_Negative (short, highly redundant) is sampled below its
# natural size, and each L3 / L4 task is repeated about 2.4-3.1 times per epoch.
V2_MIXTURE = {
    "V2_User_Profile_L1_gpt56": 30,
    "V2_User_Profile_L1_Negative_gpt56": 16,
    "V2_User_Profile_L2_gpt56": 30,
    "V2_User_Profile_L3_Persona_gpt56": 5,
    "V2_User_Profile_L3_Commercial_gpt56": 7,
    "V2_User_Profile_L4_Biography_gpt56": 5,
    "V2_User_Profile_L4_CommercialPreference_gpt56": 4,
    "V2_User_Profile_L4_Hyper_gpt56": 3,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V2_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V2_MIXTURE.values()),
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
