import pandas as pd

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer


class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        target_features = features_df.loc[:, ~features_df.columns.isin(
            ['reason_labels', 'explain_input_ids', 'explain_attention_mask'])].to_dict('records')
        explain_features = features_df.loc[:,
                           ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'reason_labels': 'labels', 'explain_input_ids': 'input_ids',
                     'explain_attention_mask': 'attention_mask'}).to_dict('records')

        target_features = super().__call__(target_features, return_tensors)
        explain_features = super().__call__(explain_features, return_tensors)

        return {
            'target': target_features,
            'explain': explain_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale

    def compute_loss(self, model, inputs, return_outputs=False):
        target_outputs = model(**inputs['target'])
        explain_outputs = model(**inputs['explain'])

        loss = self.alpha * target_outputs.loss + (1. - self.alpha) * explain_outputs.loss

        return (loss, {'target': target_outputs, 'explain': explain_outputs}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):

        target_outputs = super().prediction_step(model, inputs['target'], prediction_loss_only=False,
                                                 ignore_keys=ignore_keys)
        if self.output_rationale:
            explain_outputs = super().prediction_step(model, inputs['explain'], prediction_loss_only=False,
                                                      ignore_keys=ignore_keys)
        else:
            # Placeholder
            explain_outputs = target_outputs

        loss = self.alpha * target_outputs[0] + (1 - self.alpha) * explain_outputs[0]

        return (
            loss,
            [target_outputs[1], explain_outputs[1]],
            [target_outputs[2], explain_outputs[2]],
        )
