from transformers import LlamaModel, LlamaConfig, DynamicCache, LlavaForConditionalGeneration
from transformers.models.llama import modeling_llama
from copy import deepcopy
import torch


class HunyuanVideoLLMEncoder(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        self.auto_offload = True

    def _build_causal_mask(self, attention_mask, inputs_embeds, cache_position, position_ids, past_key_values):
        if hasattr(modeling_llama, "create_causal_mask"):
            return modeling_llama.create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        if hasattr(self, "_update_causal_mask"):
            return self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, False)

        seq_len = inputs_embeds.shape[1]
        target_len = past_key_values.get_seq_length() + seq_len if past_key_values is not None else seq_len
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal_mask = torch.full(
            (seq_len, target_len),
            fill_value=min_dtype,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.shape[0], 1, -1, -1)

        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :target_len].to(device=inputs_embeds.device)
            causal_mask = causal_mask.masked_fill(padding_mask == 0, min_dtype)

        return causal_mask

    def forward(self, input_ids, attention_mask, hidden_state_skip_layer=2):
        embed_tokens = deepcopy(self.embed_tokens).to(input_ids.device) if self.auto_offload else self.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        past_key_values = DynamicCache(config=self.config)

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self._build_causal_mask(attention_mask, inputs_embeds, cache_position, position_ids, past_key_values)
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        rotary_emb = deepcopy(self.rotary_emb).to(input_ids.device) if self.auto_offload else self.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        for layer_id, decoder_layer in enumerate(self.layers):
            if self.auto_offload:
                decoder_layer = deepcopy(decoder_layer).to(hidden_states.device)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            if layer_id + hidden_state_skip_layer + 1 >= len(self.layers):
                break

        return hidden_states


class HunyuanVideoMLLMEncoder(LlavaForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.auto_offload = False

    def enable_auto_offload(self, **kwargs):
        self.auto_offload = True

    # TODO: implement the low VRAM inference for MLLM.
    def forward(self, input_ids, pixel_values, attention_mask, hidden_state_skip_layer=2):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True,
                                  pixel_values=pixel_values)
        hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
        return hidden_state
