# GSM8K, served configuration (overlay r10), 2026-11-22

One boot of the r10 served settings (KV pool 818,388 tokens). lm-eval 0.4.12 `local-chat-completions`, GSM8K 5-shot,
first 200 problems, chat template, temperature 0, `reasoning_effort` none, concurrency 4, `max_gen_toks` 4096 and a stop
string that never occurs (`<|inkling-eval-no-stop|>`). The model reasons briefly even at the lowest effort; with
lm-eval's default budget and stop strings the answer is cut off (the r8 and r10 quality notes). Here no answer is empty.

Score: 95.5 % flexible extract (standard error 1.5), 63.5 % strict match. Strict match expects the 5-shot "#### N" answer
format, which the model's chat answers often do not follow. The local package path in `config_source` is shortened
to `<site-packages>`.
