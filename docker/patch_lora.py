"""Opt in to Inkling2's inactive-adapter and unadapted-head paths after r10.

Usage: python3 patch_lora.py /path/to/site-packages/vllm
"""

import ast
import hashlib
import sys
from pathlib import Path

LORA = "models/inkling2/nvidia/lora.py"
MODEL = "models/inkling2/nvidia/model.py"

EXPECT = {
    LORA: "bc1ac7a0415a771b5e9c76b3a26dc84799acab4e366263080c2cc129a4822954",
    MODEL: "df5f6158d5f06e77457f034519c49d5898b4aef7252049caa230b5b60b8948c5",
}

EDITS = {
    LORA: [
        (
            "from collections.abc import Iterable\n",
            "import os\nfrom collections.abc import Iterable\n",
        ),
        (
            "def add_linear_lora(\n",
            '_SKIP_INACTIVE_LORA = os.environ.get("VLLM_INKLING_LORA_SKIP") == "1"\n\n\n'
            "def add_linear_lora(\n",
        ),
        (
            "    if not isinstance(layer, BaseLayerWithLoRA):\n"
            "        return\n"
            "    if slices is None:\n",
            "    if not isinstance(layer, BaseLayerWithLoRA):\n"
            "        return\n"
            "    if _SKIP_INACTIVE_LORA:\n"
            "        # Punica refreshes this CPU flag with the token mapping, including\n"
            "        # before each zero/positive CUDA graph capture.\n"
            "        token_meta = getattr(\n"
            '            layer.punica_wrapper, "token_mapping_meta", None\n'
            "        )\n"
            '        no_lora = getattr(token_meta, "no_lora_flag_cpu", None)\n'
            "        if (\n"
            "            no_lora is not None\n"
            '            and no_lora.device.type == "cpu"\n'
            "            and no_lora.item() is True\n"
            "        ):\n"
            "            return\n"
            "    if slices is None:\n",
        ),
    ],
    MODEL: [
        (
            "from collections import Counter\n",
            "import os\nfrom collections import Counter\n",
        ),
        (
            "        processor = (\n"
            "            Inkling2LogitsProcessor\n"
            "            if vllm_config.lora_config is None\n"
            "            else LogitsProcessor\n"
            "        )\n",
            "        lora_config = vllm_config.lora_config\n"
            "        processor = (\n"
            "            Inkling2LogitsProcessor if lora_config is None else LogitsProcessor\n"
            "        )\n"
            "        if (\n"
            "            lora_config is not None\n"
            '            and os.environ.get("VLLM_INKLING_LORA_HEAD") == "1"\n'
            "        ):\n"
            "            targets = lora_config.target_modules\n"
            "            # Match the batch-sharded sampling guard in VllmConfig.\n"
            "            if targets is not None and not any(\n"
            '                target == "lm_head" or target.endswith(".lm_head") for target in targets\n'
            "            ):\n"
            "                processor = Inkling2LogitsProcessor\n",
        ),
    ],
}


def apply(root: Path) -> None:
    changes = {}
    for filename, expected in EXPECT.items():
        path = root / filename
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} != r10 input {expected}")
        source = data.decode()
        for old, new in EDITS[filename]:
            count = source.count(old)
            if count != 1:
                raise ValueError(f"{path}: anchor matched {count} times: {old[:90]!r}")
            source = source.replace(old, new, 1)
        ast.parse(source, filename=str(path))
        changes[path] = source.encode()

    for path, data in changes.items():
        path.write_bytes(data)
        print(f"lora: patched {path}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        apply(Path(sys.argv[1]))
    except (ValueError, OSError, SyntaxError) as exc:
        sys.exit(f"lora: {exc}")


if __name__ == "__main__":
    main()
