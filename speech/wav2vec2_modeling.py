import transformers
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import XVectorOutput, Wav2Vec2BaseModelOutput
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torchaudio

_HIDDEN_STATES_START_POSITION = 2


class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(
            torch.randn(input_dim, num_labels), requires_grad=True
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = (
            config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        )
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):
    def __init__(self, config, output_dim):
        super().__init__(config)
        if output_dim != None:
            config.xvector_output_dim = output_dim
        self.wav2vec2 = Wav2Vec2Model(config)

        num_layers = (
            config.num_hidden_layers + 1
        )  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.feature_extractor = nn.Linear(
            config.tdnn_dim[-1] * 2, config.xvector_output_dim
        )
        self.classifier = nn.Linear(
            config.xvector_output_dim, config.xvector_output_dim
        )

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        # output_hidden_states = True
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print("ABC")
        # print("BBB", outputs.last_hidden_state)

        # print(len(outputs.hidden_states[12]))

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
            # print("AAA", hidden_states.size())

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            tdnn_output_lengths = self._get_tdnn_output_lengths(
                feat_extract_output_lengths
            )
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)

            # print("statistic")
            # print(hidden_states.size())
            # print(mean_features.size())
            # print(std_features.size())
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        # print(statistic_pooling.size())

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        if not return_dict:
            output = (logits, output_embeddings) + outputs[
                _HIDDEN_STATES_START_POSITION:
            ]
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AudioEncoder(PreTrainedModel):

    # Is __init__ called even after PreTrainedModel.from_pretrained?
    def __init__(self, config: PretrainedConfig, output_dim=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.wav2vec2_xvector = Wav2Vec2ForXVector.from_pretrained(
            "facebook/wav2vec2-base", output_dim
        )

    def forward(self, input_values_dict):
        system_input = input_values_dict["system"]
        system_output = self.wav2vec2_xvector(
            system_input.input_values.squeeze(),
            system_input.attention_mask,
            # system_input.output_attentions,
            # system_input.output_hidden_states,
            # system_input.return_dict,
        )

        user_input = input_values_dict["user"]
        user_output = self.wav2vec2_xvector(
            user_input.input_values.squeeze(),
            user_input.attention_mask,
            # user_input.output_attentions,
            # user_input.output_hidden_states,
            # user_input.return_dict,
        )

        return (system_output, user_output)


@dataclass
class PairedAudioData:
    input_values: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor] = None
    output_attentions: Optional[bool] = None
    output_hidden_states: Optional[bool] = None
    return_dict: Optional[bool] = None


# TODO: build attention mask, ...
def build_paired_audio(system_path, user_path, processor):

    input_values_dict = {}
    # At first turn, there is no system utterence.
    # Thus we use user utterence instead. (2x user utterence)
    if system_path == None:
        system_path = user_path
    for key, path in zip(("system", "user"), (system_path, user_path)):
        speech, sample_rate = torchaudio.load(str(path))
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000).forward(speech)
        speech = speech[0][:320000]

        input_values = processor.feature_extractor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values[0]
        input_values_dict[key] = PairedAudioData(input_values=input_values)

    return input_values_dict


if __name__ == "__main__":
    device = "cpu"
    config = PretrainedConfig()
    model = AudioEncoder(config).to(device)
    # model = Wav2Vec2ForXVector.from_pretrained("facebook/wav2vec2-base").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # We may need DataCollator for batch training.
    input_values_dict = build_paired_audio(
        system_path="/home/lee1jun/develop/dev_all/MUL0032/MUL0032-0-system.wav",
        user_path="/home/lee1jun/develop/dev_all/MUL0032/MUL0032-0-user.wav",
        processor=processor,
    )

    input_values_dict = {
        "system": PairedAudioData(input_values=torch.rand(4, 1, 71147)),
        "user": PairedAudioData(input_values=torch.rand(4, 1, 61147)),
    }
    print("****")
    print(input_values_dict["system"].input_values.size())
    print(input_values_dict["user"].input_values.size())

    with torch.no_grad():
        outputs = model(input_values_dict)

    system_output = outputs[0]
    user_output = outputs[1]

    print(system_output.logits.size())
    print(user_output.logits.size())

    # logits = logits[1] # User
    # logits = logits[0]  # System
    # print(logits.embeddings.size())
    # print(logits.logits.size())

    # print(model.wav2vec2.projector)
    # print(model.wav2vec2.feature_extractor)
    # print(model.wav2vec2.classifier)
