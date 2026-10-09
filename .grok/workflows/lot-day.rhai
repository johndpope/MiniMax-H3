// lot-day — CPU contract checks for the Level-of-Token H3 tail.
//
// This track is not Separable Causal Diffusion. Agents must not import,
// edit, or launch anything under scripts/scd. This workflow does not train
// and does not write a checkpoint. The H3 weight load is a separate manual
// step, not this script.
//
// Args (optional):
//   args.minutes  number  — long-run budget after the probe passes. Default 480.
//
// Run:
//   /workflow lot-day
//   /workflow lot-day {"minutes": 480}

let meta = #{
    name: "lot-day",
    description: "CPU contract checks for the Level-of-Token H3 tail. No training and no checkpoint.",
    when_to_use: "After scripts/lot changes, before any H3 forward",
    phases: [
        #{ title: "Sanity", detail: "CPU contract tests and the tiny GPU smoke. No checkpoint." },
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
sanity_prompt += "Run these three commands yourself and wait until each exits. Do not start train_synth.py. Do not write a checkpoint.\n";
sanity_prompt += "1. python3 scripts/lot/sanity.py\n";
sanity_prompt += "2. python3 scripts/lot/test_lot.py\n";
sanity_prompt += "3. python3 scripts/lot/gpu_smoke.py\n\n";
sanity_prompt += "Set ok=true only if all three exit 0, sanity prints a line starting with 'SANITY ok ', the test script prints a line starting with 'ok ', and the smoke line contains 'finite True'.\n";
sanity_prompt += "exit_code is 0 only in that case. stdout_tail is the last 40 lines of the three commands.\n";
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

log("Sanity passed. No training and no checkpoint.");
complete(#{
    ok: true,
    summary: "CPU contract checks passed. The long synthetic run is not part of this workflow.",
});
