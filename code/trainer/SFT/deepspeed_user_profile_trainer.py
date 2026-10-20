import json

from deepspeed_llm_trainer import main
import user_profile_rules


RANKER_CONFIG = "V2_User_Profile_Ranker"

# V2 configs and their relative share of the rows of every epoch, for the HF train splits of 2026-10-19 (1,708,545
# rows: L1 267,810, L1 Negative 557,734, L2 259,918, L3 Persona 77,312, L3 Commercial 145,277, L4 Biography 161,793,
# L4 Commercial Preference 141,250, L4 Hyper 94,514, Ranker 2,937; every row <= 25,600 Qwen3.5-4B tokens).
# The weights sum to 100.7, so a share is weight / 100.7. Every profile config is seen ~1.27 times per epoch, except
# L1_Negative (short, highly redundant), which is sampled at a third of that (~0.42 passes). Ranker (0.7 -> ~11.9k
# rows, ~4 passes) is small but each pass shows its candidates in a new random order (shuffle_ranker_candidates), so
# repeats are not verbatim; more passes would start to memorize its 2,937 labels.
V2_MIXTURE = {
    "V2_User_Profile_L1_gpt56": 20.1,
    "V2_User_Profile_L1_Negative_gpt56": 13.9,
    "V2_User_Profile_L2_gpt56": 19.5,
    "V2_User_Profile_L3_Persona_gpt56": 5.8,
    "V2_User_Profile_L3_Commercial_gpt56": 10.9,
    "V2_User_Profile_L4_Biography_gpt56": 12.1,
    "V2_User_Profile_L4_CommercialPreference_gpt56": 10.6,
    "V2_User_Profile_L4_Hyper_gpt56": 7.1,
    RANKER_CONFIG: 0.7,
}

USER_PROFILE_DEFAULTS = {
    "dataset_name": "yufan/user_profile_dataset",
    "dataset_configs": list(V2_MIXTURE),
    "dataset_shuffle_seed": 42,
    "dataset_mixing_weights": list(V2_MIXTURE.values()),
}


def shuffle_ranker_candidates(messages, rng):
    """Ranker augmentation: present the candidate interests in a random order, renumber idx by the new positions
    and map output_idx to the new idx, so the same ranking is asked with different idx values."""
    prompt, marker, raw_input = messages[0]["content"].rpartition(user_profile_rules.INPUT_MARKER)
    candidates = json.loads(raw_input)
    output_idx = json.loads(messages[-1]["content"])["output_idx"]
    order = rng.permutation(len(candidates))
    new_idx = {candidates[old]["idx"]: new for new, old in enumerate(order)}
    candidates = [{**candidates[old], "idx": new} for new, old in enumerate(order)]
    answer = {"output_idx": [new_idx[idx] for idx in output_idx]}
    return [
        {"role": messages[0]["role"],
         "content": prompt + marker + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))},
        {"role": messages[-1]["role"], "content": json.dumps(answer, separators=(",", ":"))},
    ]


if __name__ == "__main__":
    main(USER_PROFILE_DEFAULTS, rollout_scorer=user_profile_rules,
         train_augmenters={RANKER_CONFIG: shuffle_ranker_candidates})
