import threading

import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)

from education_assistant import (
    EducationAssistant,
    MAX_INPUT_TOKENS,
)


class StopOnDisconnect(StoppingCriteria):
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.stop_event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


class CallbackStreamer(TextStreamer):
    def __init__(self, tokenizer, callback):
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.callback = callback

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.callback(text)


class StreamingEducationAssistant(EducationAssistant):
    def __init__(self):
        super().__init__()
        self._stream_callback = None
        self._stop_event = None

    def answer_stream(
        self,
        subject,
        question,
        passage,
        callback,
        stop_event,
    ):
        # API locking ensures only one request uses these fields.
        self._stream_callback = callback
        self._stop_event = stop_event

        try:
            return self.answer(subject, question, passage)
        finally:
            self._stream_callback = None
            self._stop_event = None

    def generate_answer(self, instruction, user_text, token_limit):
        if self._stream_callback is None:
            return super().generate_answer(
                instruction,
                user_text,
                token_limit,
            )

        prompt = self.make_prompt(instruction, user_text)

        inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_length = inputs["input_ids"].shape[1]

        if prompt_length > MAX_INPUT_TOKENS:
            raise ValueError(
                "Input is too long. Shorten the question."
            )

        stop_event = self._stop_event or threading.Event()

        if stop_event.is_set():
            return ""

        streamer = CallbackStreamer(
            self.tokenizer,
            self._stream_callback,
        )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([
                    StopOnDisconnect(stop_event),
                ]),
            )

        generated = output[0, prompt_length:]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        if (
            len(generated) == token_limit
            and not stop_event.is_set()
        ):
            warning = (
                "\n\n[The response reached its length limit "
                "and may be incomplete.]"
            )
            answer += warning
            self._stream_callback(warning)

        return answer