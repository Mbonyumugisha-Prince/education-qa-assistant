# SciQ Data Preparation

Source: https://huggingface.co/datasets/allenai/sciq

Original splits:
- Training: 11,679
- Validation: 1,000
- Test: 1,000

Preparation:
- Removed examples without supporting passages.
- Normalized whitespace.
- Preserved question, context, answer, source, and record ID.
- Preserved the original validation and test splits.
- Removed training questions matching validation/test questions.
- Removed training passages matching validation/test passages.
- Removed repeated training questions, keeping the first occurrence.

Final splits:
- Training: 10,397
- Validation: 887
- Test: 884

Verification:
- No repeated normalized questions within splits.
- No identical normalized questions across splits.
- No identical normalized supporting passages across splits.

Limitations:
- Checks do not detect all paraphrases or similar passages.
- Dataset answers are generally short, not full explanations.
- Factual accuracy has not been fully reviewed.
- This dataset supplies science coverage only.