from deepspeed_llm_trainer import main
import user_profile_rules


USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": [
        "V1_User_Profile_L1_gpt54",
        "V1_User_Profile_L2_gpt54",
        "V1_User_Profile_L3_Persona_gpt54",
        "V1_User_Profile_L3_Commercial_gpt54",
        "V1_User_Profile_L4_Biography_gpt54",
        "V1_User_Profile_L4_CommercialPreference_gpt54",
        "V1_User_Profile_L4_MissionDiscovery_gpt54",
        "V1_User_Profile_L4_MissionEnhancement_gpt54",
    ],
    "dataset_shuffle_seed": 42,
    "dataset_mixing_alpha": 0.5,
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
