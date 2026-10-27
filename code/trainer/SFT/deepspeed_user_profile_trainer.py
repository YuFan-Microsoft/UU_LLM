from deepspeed_llm_trainer import main
import user_profile_rules


# V35 configs (GPT-5.6 F7 teacher data) and their share of the rows of every epoch (%), for the HF train splits of
# 2026-10-26 (1,915,345 rows: L1 285,934, L1 Negative 557,734, L2 257,835, L3 Persona 161,761, L3 Commercial 207,217,
# L4 Biography 161,198, L4 Commercial Preference 150,080, L4 Hyper 133,586). The shares are the V3 ones. The loss is a
# token mean, so a config's share of the gradient follows its share of answer tokens, not of rows. Passes per epoch:
# L1 1.41, L2 1.49 (both run every window at inference, and L2 errors carry over), L3 Commercial 1.29 (its predicted
# queries drive behavior retrieval; ~24% of answer tokens), L4 Hyper 2.01 (the hardest task, ~6% of answer tokens at
# the natural mix), L3 Persona 0.95 (long answers already give it ~16% of answer tokens), L4 Biography 1.19 and L4
# Commercial Preference 1.15 (long inputs, short answers), and L1 Negative 0.14 (short, highly redundant, not used at
# inference).
V35_MIXTURE = {
    "V35_User_Profile_L1_gpt56": 21.0,
    "V35_User_Profile_L1_Negative_gpt56": 4.0,
    "V35_User_Profile_L2_gpt56": 20.0,
    "V35_User_Profile_L3_Persona_gpt56": 8.0,
    "V35_User_Profile_L3_Commercial_gpt56": 14.0,
    "V35_User_Profile_L4_Biography_gpt56": 10.0,
    "V35_User_Profile_L4_CommercialPreference_gpt56": 9.0,
    "V35_User_Profile_L4_Hyper_gpt56": 14.0,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V35_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V35_MIXTURE.values()),
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
