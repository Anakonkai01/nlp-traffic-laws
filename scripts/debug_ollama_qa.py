import argparse
import sys
import time

sys.path.insert(0, "src")

from config import OLLAMA_MODEL
from generate_qa import SYSTEM_PROMPT_PENALTY, call_ollama


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-predict", type=int, default=160)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--num-gpu", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    chunk = (
        "Điều 7. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với "
        "người điều khiển xe thực hiện hành vi không chấp hành hiệu lệnh "
        "của đèn tín hiệu giao thông."
    )
    start = time.time()
    result = call_ollama(
        chunk,
        SYSTEM_PROMPT_PENALTY,
        args.model,
        timeout=args.timeout,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        num_gpu=args.num_gpu,
        debug=args.debug,
    )
    elapsed = time.time() - start
    print(f"elapsed_seconds={elapsed:.2f}")
    print(result)


if __name__ == "__main__":
    main()
