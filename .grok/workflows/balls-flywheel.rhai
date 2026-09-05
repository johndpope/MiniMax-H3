let meta = #{
    name: "balls-flywheel",
    description: "Balls-mode ingest of H3 VFM flywheel graph + next highest-leverage experiment",
    when_to_use: "/balls /flywheel after research_vfm.py is running or finished. Human graph + recreate: MiniMax-H3-vfm/scripts/vfm/README.md",
    phases: [
        #{ title: "Ingest", detail: "read graph.jsonl, status.json, latest metrics and previews" },
        #{ title: "Gaps", detail: "independent balls: invert, visual, black-basin, GPU" },
        #{ title: "Next", detail: "highest-leverage next node with confidence" },
        #{ title: "Fix", detail: "edit missing loss math when visual collapsed but MSE dropped" },
        #{ title: "Execute", detail: "launch_next.sh starts that action if GPU is free" },
    ],
};

let ingest_schema = #{
    "type": "object",
    "required": ["status", "last_stage", "oracle", "invert", "v4", "nfe4", "notes"],
    "properties": #{
        "status": #{ "type": "string" },
        "last_stage": #{ "type": "string" },
        "oracle": #{ "type": "string" },
        "invert": #{ "type": "string" },
        "v4": #{ "type": "string" },
        "nfe4": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

let gaps_schema = #{
    "type": "object",
    "required": ["balls", "weakest", "visual", "basin"],
    "properties": #{
        "balls": #{ "type": "string" },
        "weakest": #{ "type": "string" },
        "visual": #{ "type": "string" },
        "basin": #{ "type": "string" },
    },
};

let next_schema = #{
    "type": "object",
    "required": ["action", "claim", "confidence", "reason"],
    "properties": #{
        "action": #{ "type": "string" },
        "claim": #{ "type": "string" },
        "confidence": #{ "type": "number" },
        "reason": #{ "type": "string" },
    },
};

let fix_schema = #{
    "type": "object",
    "required": ["edited", "notes"],
    "properties": #{
        "edited": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let exec_schema = #{
    "type": "object",
    "required": ["started", "cmd", "notes"],
    "properties": #{
        "started": #{ "type": "boolean" },
        "cmd": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

let run_dir = if args == () { () } else { args.run_dir };
if run_dir == () {
    pause("verification", "Pass args.run_dir — MiniMax-H3-vfm/scripts/vfm/runs/flywheel");
}

phase("Ingest");
let p_in = "Read with read_file/list_dir, do not guess: ";
p_in += run_dir + "/status.json, " + run_dir + "/graph.jsonl, " + run_dir + "/next_action.json. ";
p_in += "Also " + run_dir + "/oracle/metrics.jsonl, invert/summary.json, v4/metrics.jsonl, nfe4/summary.json, ";
p_in += "lora/metrics.jsonl, i2v/metrics.jsonl, lora_gauss_collapse/metrics.jsonl if they exist. ";
p_in += "List and read the latest PNG under i2v/live.png, i2v/live_first.png, i2v/previews, lora/previews, invert/previews. ";
p_in += "status is running/finished/crashed/missing. last_stage is the latest experiment id. ";
p_in += "oracle/invert/v4/nfe4 are pass, fail, running, or missing. notes must quote numbers from files. ";
p_in += "A Traceback in " + run_dir + "/train.log means crashed. A dirty reboot with no SUMMARY is crashed.";
let ing = agent(p_in, #{
    label: "ingest",
    capability_mode: "read-only",
    output_schema: ingest_schema,
});

phase("Gaps");
let p_g = "Using only facts from these paths (re-read if needed): " + run_dir;
p_g += ". Run Balls on five independent questions: ";
p_g += "(1) Does some z exist for frozen 1-NFE (invert vs Gaussian)? ";
p_g += "(2) Do preview PNGs show a scene vs speckle vs black? Read the latest PNG. Left panel is GT, middle is the method. ";
p_g += "(3) If recon dropped while RGB std died or the middle panel went black, that is the v=-z basin, not a PASS. ";
p_g += "(4) Is GPU research_vfm.py alive? ";
p_g += "(5) Is vfm_nfe_loss (cosine+std+obs+grad) already in train_h3_vfm.py? ";
p_g += "visual is scene, speckle, black, or unknown. basin is none, v_neg_z, or unknown. ";
p_g += "balls is a markdown table Ball | Answer | Confidence | Notes. weakest is the lowest-confidence ball that still affects the next action. ";
p_g += "Do not inflate confidence. Visual structure requires looking at a PNG, not MSE.";
let gaps = agent(p_g, #{
    label: "gaps",
    capability_mode: "read-only",
    output_schema: gaps_schema,
});

phase("Next");
let p_n = "Propose the single highest-leverage next experiment. ";
p_n += "The user already granted invent + GPU. Do NOT pick wait because you need permission to change a loss. ";
p_n += "Allowed tokens: wait, stop, i2v, lora_mix, fix_loss. ";
p_n += "i2v = official VFM on I2VA (first-frame y, Gaussian adapter, joint LoRA, 80/20 mix). Prefer i2v after lora_slow PASS. Live recon at i2v/live.png. ";
p_n += "fix_loss = MSE improved but visual is black/speckle, OR vfm_nfe_loss is missing. You will edit code in the next phase, then launch i2v (not another mix crawl). ";
p_n += "lora_mix = DO NOT pick; mix already PASS at lora_step00750. ";
p_n += "wait = research_vfm.py is alive (GPU busy). ";
p_n += "stop = i2v first-frame live.png is a recognizable street/face AND mid is not collapsed. ";
p_n += "Never pick pin, v4, zfit, invert, nfe4, lora_dit, lora_mix. Those nodes are closed. ";
p_n += "Settled: frozen 1-NFE(z★) is a face; v1b cannot emit z★; latent L2 ≡ ||v-(x0-z)|| and prefers v=-z (black). ";
p_n += "action is one token. claim is one sentence. confidence 0-1. reason cites files.";
let nxt = agent(p_n, #{
    label: "next-node",
    capability_mode: "read-only",
    output_schema: next_schema,
});

let status = "unknown";
let last = "";
if ing != () && ing.success && ing.output.status != () {
    status = ing.output.status;
    last = ing.output.last_stage;
}
let action = "wait";
let claim = "";
let conf = 0.0;
if nxt != () && nxt.success {
    action = nxt.output.action;
    claim = nxt.output.claim;
    conf = nxt.output.confidence;
}
let balls = "";
let visual = "unknown";
let basin = "unknown";
if gaps != () && gaps.success {
    balls = gaps.output.balls;
    if gaps.output.visual != () { visual = gaps.output.visual; }
    if gaps.output.basin != () { basin = gaps.output.basin; }
}
if (visual == "black" || basin == "v_neg_z") && action != "wait" && action != "stop" {
    action = "fix_loss";
}

phase("Fix");
let fix_notes = "skipped";
if action == "fix_loss" {
    let p_f = "You have standing permission to change the loss. Do it. Do not ask the user. ";
    p_f += "Read MiniMax-H3-vfm/scripts/vfm/train_h3_vfm.py and research_vfm.py. ";
    p_f += "If vfm_nfe_loss is missing, add it: L_MF down-weighted, plus 1-cos(xhat,x0), (std(xhat)-std(x0))^2, ";
    p_f += "latent-RGB L_obs (no clamp), mid-frame nabla RGB, and lora-mix homotopy z=(1-a)zstar+a*eps with a=(i/N)^2. ";
    p_f += "Why: ||v-(x0-z)||^2 ≡ ||xhat-x0||^2 and v=-z (black) beats speckle under plain MSE. ";
    p_f += "Wire run_lora to vfm_nfe_loss. launch_next.sh lora_mix must pass --lora-mix and the weights. ";
    p_f += "Do not empty_cache on this display GPU. Do not relaunch v4/zfit/pin. ";
    p_f += "edited=true only if you wrote files or confirmed vfm_nfe_loss already exists. notes quote the identity and weights.";
    let fx = agent(p_f, #{
        label: "fix-loss",
        capability_mode: "read-write",
        output_schema: fix_schema,
    });
    if fx != () && fx.success {
        fix_notes = fx.output.notes;
        action = "i2v";
    } else {
        fix_notes = "fix agent failed; still try i2v if code already has vfm_nfe_loss";
        action = "i2v";
    }
}

phase("Execute");
let started = false;
let exec_notes = "skipped";
if action == "wait" || action == "stop" {
    exec_notes = "no launch for " + action;
} else {
    let launch_tok = action;
    if launch_tok == "fix_loss" { launch_tok = "i2v"; }
    let p_e = "Take action. Do not wait for permission. Write " + run_dir + "/next_action.json as ";
    p_e += "{\"action\":\"" + launch_tok + "\"} using a shell redirect. Then run: ";
    p_e += "chmod +x /home/johndpope/Documents/GitHub/MiniMax-H3-vfm/scripts/vfm/launch_next.sh && ";
    p_e += "nohup /home/johndpope/Documents/GitHub/MiniMax-H3-vfm/scripts/vfm/launch_next.sh " + launch_tok;
    p_e += " >> " + run_dir + "/launch_next.log 2>&1 & ";
    p_e += "echo STARTED $! . If GPU is busy the script exits 0 without starting a second job. ";
    p_e += "started=true only if you launched or confirmed research_vfm.py is running that action. ";
    p_e += "cmd is the command you ran. notes quote launch_next stdout.";
    let ex = agent(p_e, #{
        label: "execute",
        capability_mode: "execute",
        output_schema: exec_schema,
    });
    if ex != () && ex.success {
        started = ex.output.started;
        exec_notes = ex.output.notes;
    }
}

let report = "# balls-flywheel\n\n";
report += "- run_dir: " + run_dir + "\n";
report += "- status: " + status + "\n";
report += "- last_stage: " + last + "\n";
report += "- visual: " + visual + "\n";
report += "- basin: " + basin + "\n";
report += "- next_action: " + action + "\n";
report += "- executed: " + started.to_string() + "\n";
report += "- confidence: " + conf.to_string() + "\n";
report += "- claim: " + claim + "\n";
report += "- fix: " + fix_notes + "\n";
report += "- exec: " + exec_notes + "\n\n";
report += balls + "\n";
let path = write_scratch_file("balls_flywheel.md", report);
log("action=" + action + " started=" + started.to_string() + " visual=" + visual);
complete(#{ path: path, action: action, started: started, status: status, last_stage: last, confidence: conf, visual: visual });
