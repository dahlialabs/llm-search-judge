#!/bin/bash

N="${1:-20000}"


poetry run python -m local_llm_judge.main --verbose --eval-fn brand2 --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn image_embedding --simplify-query --N $N

poetry run python -m local_llm_judge.main --verbose --eval-fn name --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn brand --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn brand2 --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn desc --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn category --check-both-ways --N $N
poetry run python -m local_llm_judge.main --verbose --eval-fn captions --check-both-ways --N $N

# poetry run python -m local_llm_judge.main --verbose --eval-fn all_fields --check-both-ways --N $N

# poetry run python -m local_llm_judge.main --verbose --eval-fn name --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn desc --N $N

# poetry run python -m local_llm_judge.main --verbose --eval-fn all_fields_allow_neither --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn all_fields --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn all_fields_allow_neither --check-both-ways --N $N

# poetry run python -m local_llm_judge.main --verbose --eval-fn category_allow_neither --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn category --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn category_allow_neither --check-both-ways --N $N

# poetry run python -m local_llm_judge.main --verbose --eval-fn name_allow_neither --check-both-ways --N $N
#poetry run python -m local_llm_judge.main --verbose --eval-fn name_allow_neither --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn desc_allow_neither --check-both-ways --N $N
# poetry run python -m local_llm_judge.main --verbose --eval-fn desc_allow_neither --N $N
