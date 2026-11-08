#!/usr/bin/env python3
"""node-0's full training log in the W&B Logs tab: train.log (the launcher's tee of the verl driver, which prints the
output of every Ray worker) streamed into the run the trainer creates.

The launcher picks the run id (+trainer.wandb_run_id), starts this process in the background with its stdout to
/dev/null, then runs the training. The trainer (verl/utils/tracking.py) creates (or resumes) the run in W&B shared
mode as the primary process, without console capture (its console is part of train.log); this process waits until
the run is running, joins it as a non-primary process labelled --label (node-0) with console capture, and copies
train.log, from its first line, to its own stdout, so every line reaches the Logs tab. When the launcher exits it
copies what is left and leaves (a non-primary process does not mark the run finished). Its stdout is never part of
train.log, so no line comes back. Any W&B problem only ends this process (written to its stderr); the training is
not affected.

    python3 log_to_wandb.py --run_id <id> --project <project> --log <train.log> --parent_pid $$ > /dev/null 2> <err> &
"""

import argparse
import os
import sys
import time


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--log", required=True, help="The launcher's train.log")
    parser.add_argument("--parent_pid", type=int, required=True, help="The launcher: stop once it has exited")
    parser.add_argument("--label", default="node-0", help="The Logs tab label of these lines")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    while True:  # the trainer starts (or resumes) the run once Ray and the workers are up (minutes)
        api.flush()  # no cached run state
        try:
            if api.run(f"{entity}/{args.project}/{args.run_id}").state == "running":
                break
        except Exception:
            pass
        if not alive(args.parent_pid):
            return
        time.sleep(15)
    with open(args.log, encoding="utf-8", errors="replace") as log:
        if "W&B shared mode " in log.read():  # verl/utils/tracking.py fell back to a run of its own: do not join it
            print("The trainer's run is not in W&B shared mode; the Logs tab keeps its console only", file=sys.stderr)
            return
    run = wandb.init(entity=entity, project=args.project, id=args.run_id,
                     dir=os.path.dirname(os.path.abspath(args.log)),  # the run's output directory, not the code
                     settings=wandb.Settings(mode="shared", x_primary=False, x_label=args.label,
                                             x_update_finish_state=False, console="redirect"))
    with open(args.log, encoding="utf-8", errors="replace", newline="") as log:
        while True:
            position = log.tell()
            line = log.readline()
            if line.endswith("\n") or (line and not alive(args.parent_pid)):
                sys.stdout.write(line)
                continue
            log.seek(position)  # nothing new, or a line still being written
            sys.stdout.flush()
            if not alive(args.parent_pid):
                break
            time.sleep(1)
    sys.stdout.flush()
    run.finish()


if __name__ == "__main__":
    main()
