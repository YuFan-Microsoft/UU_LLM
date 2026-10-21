from deepspeed_llm_trainer import main
import user_profile_rules


# V2 configs and their share of the rows of every epoch (%), for the HF train splits of 2026-10-15 (1,705,608 rows:
# L1 267,810, L1 Negative 557,734, L2 259,918, L3 Persona 77,312, L3 Commercial 145,277, L4 Biography 161,793,
# L4 Commercial Preference 141,250, L4 Hyper 94,514). Every config is seen ~1.28 times per epoch, except L1_Negative
# (short, highly redundant), which is sampled at a third of that (~0.43 passes).
V2_MIXTURE = {
    "V2_User_Profile_L1_gpt56": 20.1,
    "V2_User_Profile_L1_Negative_gpt56": 13.9,
    "V2_User_Profile_L2_gpt56": 19.5,
    "V2_User_Profile_L3_Persona_gpt56": 5.8,
    "V2_User_Profile_L3_Commercial_gpt56": 10.9,
    "V2_User_Profile_L4_Biography_gpt56": 12.1,
    "V2_User_Profile_L4_CommercialPreference_gpt56": 10.6,
    "V2_User_Profile_L4_Hyper_gpt56": 7.1,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V2_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V2_MIXTURE.values()),
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
