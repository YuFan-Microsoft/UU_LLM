from deepspeed_llm_trainer import main
import user_profile_rules


# V2 configs and their share of the rows of every epoch (%), for the HF train splits of 2026-10-15 (1,705,608 rows:
# L1 267,810, L1 Negative 557,734, L2 259,918, L3 Persona 77,312, L3 Commercial 145,277, L4 Biography 161,793,
# L4 Commercial Preference 141,250, L4 Hyper 94,514). The loss is a token mean, so a config's share of the gradient
# follows its share of answer tokens, not of rows. Passes per epoch: L1 1.34, L2 1.38 (both run every window at
# inference, and L2 errors carry over), L3 Commercial 1.76 (its predicted queries drive behavior retrieval), L4 Hyper
# 2.17 (the hardest task, ~4.6% of answer tokens at the natural mix), L3 Persona 1.10 (long answers already give it
# ~14% of answer tokens), L4 Biography 1.16 and L4 Commercial Preference 1.21 (long inputs, short answers), and
# L1 Negative 0.15 (short, highly redundant, not used at inference).
V2_MIXTURE = {
    "V2_User_Profile_L1_gpt56": 21.0,
    "V2_User_Profile_L1_Negative_gpt56": 5.0,
    "V2_User_Profile_L2_gpt56": 21.0,
    "V2_User_Profile_L3_Persona_gpt56": 5.0,
    "V2_User_Profile_L3_Commercial_gpt56": 15.0,
    "V2_User_Profile_L4_Biography_gpt56": 11.0,
    "V2_User_Profile_L4_CommercialPreference_gpt56": 10.0,
    "V2_User_Profile_L4_Hyper_gpt56": 12.0,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V2_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V2_MIXTURE.values()),
}


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules)
