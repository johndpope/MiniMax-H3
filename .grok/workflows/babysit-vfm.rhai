let meta = #{
    name: "babysit-vfm",
    description: "Inspect H3 VFM training: wandb-offline metrics, recon previews, collapse vs PASS",
    when_to_use: "After research_vfm.py / train_h3_vfm.py is running or finished; diagnose losses and previews. Recreate commands: MiniMax-H3-vfm/scripts/vfm/README.md",
    phases: [
        #{ title: "Logs", detail: "read train.log, metrics.jsonl, smoke.json" },
        #{ title: "Previews", detail: "inspect reconstruction strips" },
        #{ title: "Verdict", detail: "PASS/FAIL/collapse with evidence" },
    ],
};

let logs_schema = #{
    "type": "object",
    "required": ["status", "latest_step", "recon_v", "base_recon_v", "mu_norm", "notes"],
    "properties": #{
        "status": #{ "type": "string" },
        "latest_step": #{ "type": "integer" },
        "recon_v": #{ "type": "number" },
        "base_recon_v": #{ "type": "number" },
        "mu_norm": #{ "type": "number" },
        "notes": #{ "type": "string" },
    },
};

let preview_schema = #{
    "type": "object",
    "required": ["n_previews", "structure", "notes"],
    "properties": #{
        "n_previews": #{ "type": "integer" },
        "structure": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

let verdict_schema = #{
    "type": "object",
    "required": ["verdict", "reason", "evidence"],
    "properties": #{
        "verdict": #{ "type": "string" },
        "reason": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let run_dir = if args == () { () } else { args.run_dir };
if run_dir == () {
    pause("verification", "Pass args.run_dir — path to scripts/vfm/runs/real_h3");
}

phase("Logs");
let p_logs = "Read these files with the read_file tool (do not guess): ";
p_logs += run_dir + "/train.log, " + run_dir + "/metrics.jsonl, " + run_dir + "/smoke.json. ";
p_logs += "If a file is missing, say so. Extract latest_step, recon_v, base_recon_v, mu_norm, kl_v. ";
p_logs += "status is one of: running, crashed, finished. notes must quote numbers from the files. ";
p_logs += "An empty metrics.jsonl with a Traceback in train.log is crashed, not running.";
let logs = agent(p_logs, #{
    label: "vfm-logs",
    capability_mode: "read-only",
    output_schema: logs_schema,
});

phase("Previews");
let p_prev = "List and read PNG previews under " + run_dir + "/previews/ using list_dir and read_file. ";
p_prev += "Prefer " + run_dir + "/i2v/live.png and i2v/live_first.png if present (8x nearest, updating). ";
p_prev += "I2V four-panel is y=first-frame | 1-NFE first | GT mid | 1-NFE mid. Judge panel 2 (first-frame recon) and panel 4 (mid). ";
p_prev += "Older strips are three panels left-to-right: ground-truth latent-RGB, method 1-NFE, Gaussian/second sample. ";
p_prev += "The FACE on the left is GT. Judge the MIDDLE panel only for 3-panel strips. ";
p_prev += "n_previews is the count of PNG files. structure is: none, noise, black, or scene. ";
p_prev += "black = middle panel near-zero RGB std vs GT. notes must describe the middle panel. Empty folder: n_previews=0 structure=none.";
let prev = agent(p_prev, #{
    label: "vfm-previews",
    capability_mode: "read-only",
    output_schema: preview_schema,
});

phase("Verdict");
let p_ver = "Decide a verdict for the H3 VFM run. Read the same files if you need: " + run_dir;
p_ver += ". Verdict must be one of: pass, fail_collapse, fail_mu_collapse, fail_recon, crashed, too_early. ";
p_ver += "pass only if recon beat Gaussian AND the MIDDLE preview panel is a recognizable scene (street/face), not GT-left-only. ";
p_ver += "fail_collapse if recon dropped but the middle panel is black/speckle (v=-z basin: ||xhat-x0|| prefers xhat=0). This is not a PASS. ";
p_ver += "fail_mu_collapse if ||mu|| ~ 0. fail_recon if training finished and did not beat Gaussian. ";
p_ver += "crashed if train.log has Traceback/OOM or the box dirty-rebooted mid-run. too_early if fewer than 5 logged steps. ";
p_ver += "reason and evidence must cite concrete numbers or image observations. Default away from pass.";
let verd = agent(p_ver, #{
    label: "vfm-verdict",
    capability_mode: "read-only",
    output_schema: verdict_schema,
});

let status = "unknown";
let latest = 0;
let recon = 0.0;
let base = 0.0;
if logs != () && logs.success && logs.output.status != () {
    status = logs.output.status;
    latest = logs.output.latest_step;
    recon = logs.output.recon_v;
    base = logs.output.base_recon_v;
}
let nprev = 0;
if prev != () && prev.success { nprev = prev.output.n_previews; }
let verdict = "unknown";
let reason = "";
if verd != () && verd.success {
    verdict = verd.output.verdict;
    reason = verd.output.reason;
}

let report = "# VFM babysit\n\n";
report += "- run_dir: " + run_dir + "\n";
report += "- status: " + status + "\n";
report += "- latest_step: " + latest.to_string() + "\n";
report += "- recon_v: " + recon.to_string() + " base: " + base.to_string() + "\n";
report += "- previews: " + nprev.to_string() + "\n";
report += "- verdict: " + verdict + "\n";
report += "- reason: " + reason + "\n";
let path = write_scratch_file("vfm_babysit.md", report);
log("verdict=" + verdict + " step=" + latest.to_string());
complete(#{ path: path, verdict: verdict, latest_step: latest, status: status });
