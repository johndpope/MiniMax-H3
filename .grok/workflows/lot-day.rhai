// lot-day — sanity-check Level-of-Token, then train it on the one GPU.
//
// This track is not Separable Causal Diffusion. Agents must not import,
// edit, or launch anything under scripts/scd, and must not load an H3
// checkpoint. Training is synthetic flow-matching in scripts/lot/train_synth.py.
//
// Args (optional):
//   args.minutes  number  — long-run budget after the probe passes. Default 480.
//
// Run:
//   /workflow lot-day
//   /workflow lot-day {"minutes": 480}

let meta = #{
    name: "lot-day",
    description: "Sanity-check Level-of-Token, probe-train on one GPU, then keep a long synthetic run",
    when_to_use: "After scripts/lot changes, or to spend the day training the LoT adapter on the single GPU",
    phases: [
        #{ title: "Sanity", detail: "CPU tests and GPU smoke" },
        #{ title: "Probe", detail: "200 steps, held-out loss must drop" },
        #{ title: "Train", detail: "long single-GPU run" },
        #{ title: "Review", detail: "read the curve and the checkpoint" },
    ],
};

let check_schema = #{
    "type": "object",
    "required": ["ok", "exit_code", "stdout_tail", "error"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "exit_code": #{ "type": "integer" },
        "stdout_tail": #{ "type": "string" },
        "error": #{ "type": "string" },
    },
};

let train_schema = #{
    "type": "object",
    "required": ["ok", "exit_code", "steps", "start_loss", "end_loss", "nan", "peak_mb", "out_dir", "stdout_tail", "error"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "exit_code": #{ "type": "integer" },
        "steps": #{ "type": "integer" },
        "start_loss": #{ "type": "number" },
        "end_loss": #{ "type": "number" },
        "nan": #{ "type": "integer" },
        "peak_mb": #{ "type": "number" },
        "out_dir": #{ "type": "string" },
        "stdout_tail": #{ "type": "string" },
        "error": #{ "type": "string" },
    },
};

let review_schema = #{
    "type": "object",
    "required": ["ok", "eval_dropped", "checkpoint_exists", "nan_seen", "evidence", "issues"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "eval_dropped": #{ "type": "boolean" },
        "checkpoint_exists": #{ "type": "boolean" },
        "nan_seen": #{ "type": "boolean" },
        "evidence": #{ "type": "string" },
        "issues": #{ "type": "string" },
    },
};

let minutes = 480;
if args != () {
    if args.minutes != () {
        minutes = args.minutes;
    }
}
let minute_text = minutes.to_string();

fn failed(summary) {
    complete(#{
        ok: false,
        summary: summary,
    });
}

// ---------------------------------------------------------------------
phase("Sanity");
log("Sanity: scripts/lot tests, then the GPU smoke. No SCD, no H3 weights.");

let sanity_prompt = "";
sanity_prompt += "Working directory: /home/johndpope/Documents/GitHub/MiniMax-H3\n";
sanity_prompt += "This is the Level-of-Token track. Do not open, edit, import, or run anything under scripts/scd.\n";
sanity_prompt += "Do not load an H3 checkpoint. Do not call torch.cuda.empty_cache.\n";
sanity_prompt += "Do not edit source files. Do not commit.\n\n";
sanity_prompt += "Run these two commands yourself and wait until each exits:\n";
sanity_prompt += "1. python3 scripts/lot/test_lot.py\n";
sanity_prompt += "2. python3 scripts/lot/gpu_smoke.py\n\n";
sanity_prompt += "Set ok=true only if both exit 0, the test script prints a line starting with 'ok ', and the smoke line contains 'finite True'.\n";
sanity_prompt += "exit_code is 0 only in that case. stdout_tail is the last 40 lines of both commands.\n";
sanity_prompt += "error is empty when ok is true, otherwise quote the failing line.\n";

let sanity = agent(sanity_prompt, #{
    label: "lot-sanity",
    capability_mode: "execute",
    output_schema: check_schema,
});

if sanity == () || !sanity.success || sanity.output == () || sanity.output.ok != true {
    let why = "sanity checks did not pass";
    if sanity != () && sanity.success && sanity.output != () && sanity.output.error != () {
        why = sanity.output.error;
    }
    failed(why);
}

log("Sanity passed.");

// ---------------------------------------------------------------------
phase("Probe");
log("Probe: 200 synthetic steps. Held-out eval loss must fall.");

let probe_prompt = "";
probe_prompt += "Working directory: /home/johndpope/Documents/GitHub/MiniMax-H3\n";
probe_prompt += "Level-of-Token only. Do not touch scripts/scd. Do not load an H3 checkpoint.\n";
probe_prompt += "Do not call torch.cuda.empty_cache. Do not edit source. Do not commit.\n\n";
probe_prompt += "Run and wait:\n";
probe_prompt += "python3 scripts/lot/train_synth.py --steps 200 --out scripts/lot/runs/probe --seed 0 --log-every 20 --save-every 200\n\n";
probe_prompt += "Read the final stdout line that starts with LOT_RESULT.\n";
probe_prompt += "Fill steps, start_loss, end_loss, nan, peak_mb, and out_dir from that line.\n";
probe_prompt += "Set ok=true only if the process exits 0, nan is 0, and end_loss is strictly less than start_loss.\n";
probe_prompt += "exit_code is the process exit code. error is empty on success, otherwise the traceback tail.\n";
probe_prompt += "stdout_tail must include the LOT_RESULT line.\n";

let probe = agent(probe_prompt, #{
    label: "lot-probe",
    capability_mode: "execute",
    output_schema: train_schema,
});

if probe == () || !probe.success || probe.output == () || probe.output.ok != true {
    let why = "probe training did not reduce held-out loss";
    if probe != () && probe.success && probe.output != () && probe.output.error != () {
        why = probe.output.error;
    }
    failed(why);
}

log("Probe passed. Starting the long run for " + minute_text + " minutes.");

// ---------------------------------------------------------------------
phase("Train");

let train_prompt = "";
train_prompt += "Working directory: /home/johndpope/Documents/GitHub/MiniMax-H3\n";
train_prompt += "Level-of-Token only. Do not touch scripts/scd. Do not load an H3 checkpoint.\n";
train_prompt += "Do not call torch.cuda.empty_cache. Do not edit source. Do not commit.\n";
train_prompt += "Do not start a second training process. This run is long. Wait for it to exit.\n\n";
train_prompt += "Run and wait:\n";
train_prompt += "python3 scripts/lot/train_synth.py --minutes " + minute_text;
train_prompt += " --out scripts/lot/runs/day --seed 1 --log-every 50 --save-every 2000\n\n";
train_prompt += "Read the final stdout line that starts with LOT_RESULT.\n";
train_prompt += "Fill steps, start_loss, end_loss, nan, peak_mb, and out_dir from that line.\n";
train_prompt += "Set ok=true only if the process exits 0, nan is 0, and end_loss is strictly less than start_loss.\n";
train_prompt += "stdout_tail must include the LOT_RESULT line. error is empty on success.\n";

let trained = agent(train_prompt, #{
    label: "lot-train",
    capability_mode: "execute",
    output_schema: train_schema,
});

if trained == () || !trained.success || trained.output == () {
    failed("long training agent returned no usable output");
}

// ---------------------------------------------------------------------
phase("Review");

let quoted = json_encode(trained.output);
let review_prompt = "";
review_prompt += "You are checking a finished Level-of-Token synthetic training run.\n";
review_prompt += "Repo: /home/johndpope/Documents/GitHub/MiniMax-H3\n";
review_prompt += "Do not edit files. Do not run training. Do not open scripts/scd.\n";
review_prompt += "Read these with the file tools, do not trust memory:\n";
review_prompt += "1. scripts/lot/runs/day/log.jsonl — first and last records that contain an eval field.\n";
review_prompt += "2. Confirm scripts/lot/runs/day/last.pt exists (list the directory).\n";
review_prompt += "The trainer's own LOT_RESULT, which may be wrong, is: " + quoted + "\n";
review_prompt += "Set eval_dropped=true only if the last logged eval is lower than the first logged eval.\n";
review_prompt += "Set checkpoint_exists=true only if last.pt is in that directory.\n";
review_prompt += "Set nan_seen=true if any log record has nan=true or a null loss.\n";
review_prompt += "Set ok=true only when eval_dropped is true, checkpoint_exists is true, and nan_seen is false.\n";
review_prompt += "evidence must quote the first eval, the last eval, the step numbers, and the checkpoint path.\n";
review_prompt += "issues is empty when ok is true.\n";

let review = agent(review_prompt, #{
    label: "lot-review",
    capability_mode: "read-only",
    output_schema: review_schema,
});

let summary = "Level-of-Token day run finished.";
let review_ok = false;
if review != () && review.success && review.output != () && review.output.ok == true {
    review_ok = true;
    summary = "Held-out eval fell and last.pt is on disk.";
} else {
    summary = "Long run ended but the review did not confirm a clean eval drop.";
}

let report = summary + "\n";
report += "Minutes budget: " + minute_text + "\n";
if trained.output != () {
    if trained.output.steps != () { report += "Steps: " + trained.output.steps.to_string() + "\n"; }
    if trained.output.start_loss != () { report += "Start eval: " + trained.output.start_loss.to_string() + "\n"; }
    if trained.output.end_loss != () { report += "End eval: " + trained.output.end_loss.to_string() + "\n"; }
    if trained.output.peak_mb != () { report += "Peak MB: " + trained.output.peak_mb.to_string() + "\n"; }
}
if review != () && review.success && review.output != () && review.output.evidence != () {
    report += "Review: " + review.output.evidence + "\n";
}
let report_path = write_scratch_file("lot-day-report.md", report);
log(summary);
complete(#{
    ok: review_ok,
    summary: summary,
    minutes: minutes,
    report: report_path,
});
