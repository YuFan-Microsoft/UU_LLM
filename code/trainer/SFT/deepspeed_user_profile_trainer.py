from deepspeed_llm_trainer import main
import user_profile_rules


# V3 configs and their share of the rows of every epoch (%), for the HF train splits of 2026-10-24 (1,922,110 rows:
# L1 287,557, L1 Negative 557,734, L2 259,606, L3 Persona 162,863, L3 Commercial 201,304, L4 Biography 161,345,
# L4 Commercial Preference 155,190, L4 Hyper 136,511). The loss is a token mean, so a config's share of the gradient
# follows its share of answer tokens, not of rows. Passes per epoch: L1 1.40, L2 1.48 (both run every window at
# inference, and L2 errors carry over), L3 Commercial 1.34 (its predicted queries drive behavior retrieval; ~25% of
# answer tokens), L4 Hyper 1.97 (the hardest task, ~6% of answer tokens at the natural mix), L3 Persona 0.94 (V3 has
# twice the V2 rows and long answers already give it ~15% of answer tokens), L4 Biography 1.19 and L4 Commercial
# Preference 1.11 (long inputs, short answers), and L1 Negative 0.14 (short, highly redundant, not used at inference).
V3_MIXTURE = {
    "V3_User_Profile_L1_gpt56": 21.0,
    "V3_User_Profile_L1_Negative_gpt56": 4.0,
    "V3_User_Profile_L2_gpt56": 20.0,
    "V3_User_Profile_L3_Persona_gpt56": 8.0,
    "V3_User_Profile_L3_Commercial_gpt56": 14.0,
    "V3_User_Profile_L4_Biography_gpt56": 10.0,
    "V3_User_Profile_L4_CommercialPreference_gpt56": 9.0,
    "V3_User_Profile_L4_Hyper_gpt56": 14.0,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V3_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V3_MIXTURE.values()),
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
