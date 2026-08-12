// validate-scd-arch — push the SCD (Separable Causal Diffusion) port through
// the architecture gate and produce a readable results report.
//
// What it does (three phases):
//   1. Run  scripts/scd/validate_architecture.py  (unit tests + claims + stored benches)
//   2. Read the JSON/Markdown it wrote under docs/ and double-check the gates
//   3. Write a short handoff summary into workflow scratch
//
// Args (all optional):
//   args.skip_tests  bool  — skip unit tests (claims + stored numbers only)
//   args.bench       bool  — also re-run Tier 0 GPU microbench (slow)
//   args.strict      bool  — treat check_findings warnings as failures
//
// Run:
//   /workflow validate-scd-arch
//   /workflow validate-scd-arch {"bench": true}
//
// This is a local research gate (docs/ + scripts/scd/). Do not treat it as
// something to push upstream to MiniMaxAI/MiniMax-H3.

let meta = #{
    name: "validate-scd-arch",
    description: "Run SCD architecture validation (tests + claims + speed/VRAM numbers) and write docs/ results",
    when_to_use: "After changing scd_attention / scd_model / LoRA / design-doc numbers, or before trusting speed/VRAM claims",
    phases: [
        #{ title: "Run gate", detail: "execute validate_architecture.py" },
        #{ title: "Verify outputs", detail: "read JSON + report; confirm gates" },
        #{ title: "Summarize", detail: "handoff note in scratch" },
    ],
};

// ---------- helpers ----------

fn as_bool(v) {
    // args may arrive as bool, string, or unit. Treat missing as false.
    if v == () { return false; }
    if type_of(v) == "bool" { return v; }
    if type_of(v) == "string" {
        // Accept common truthy spellings; avoid depending on case helpers.
        return v == "true" || v == "True" || v == "TRUE"
            || v == "1" || v == "yes" || v == "Yes";
    }
    false
}

fn gate_line(name, ok) {
    if ok { "PASS  " + name } else { "FAIL  " + name }
}

// ---------- args ----------

let skip_tests = false;
let do_bench = false;
let strict = false;
if args != () {
    skip_tests = as_bool(args.skip_tests);
    do_bench = as_bool(args.bench);
    strict = as_bool(args.strict);
}

// Build the shell command. Always from repo root; the script finds docs/ itself.
let cmd = "python3 scripts/scd/validate_architecture.py";
if skip_tests { cmd += " --skip-tests"; }
if do_bench { cmd += " --bench"; }
if strict { cmd += " --strict"; }

log("Architecture gate command: " + cmd);

// =====================================================================
// Phase 1 — run the validator (needs execute: subprocesses + optional GPU)
// =====================================================================
phase("Run gate");

let run_prompt = "";
run_prompt += "You are running the SCD architecture validation gate in this MiniMax-H3 repo.\n";
run_prompt += "Working directory MUST be the repo root (the folder that contains docs/ and scripts/scd/).\n\n";
run_prompt += "Do exactly this:\n";
run_prompt += "1. Confirm cwd with `pwd` and that docs/ exists.\n";
run_prompt += "2. Run this command and wait for it to finish (unit tests take ~30–90s; --bench is minutes):\n";
run_prompt += "   " + cmd + "\n";
run_prompt += "3. Do NOT edit source files. Do NOT git commit. Do NOT start training or decoding.\n";
run_prompt += "4. After the command exits, check that these exist:\n";
run_prompt += "   - docs/arch_validation_results.json\n";
run_prompt += "   - docs/arch_validation_report.md\n";
run_prompt += "5. Return a JSON object matching the schema. Put the command's exit code, overall PASS/FAIL\n";
run_prompt += "   (from the printed OVERALL line or the JSON ok field), and a short stdout tail.\n";
run_prompt += "If the command is missing or the repo root is wrong, set ok=false and explain in error.\n";

let run_schema = #{
    "type": "object",
    "required": ["ok", "exit_code", "results_json", "report_md", "stdout_tail", "error"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "exit_code": #{ "type": "integer" },
        "results_json": #{ "type": "string" },
        "report_md": #{ "type": "string" },
        "stdout_tail": #{ "type": "string" },
        "error": #{ "type": "string" },
    },
};

let run_res = agent(run_prompt, #{
    label: "run-validate-architecture",
    capability_mode: "execute",
    output_schema: run_schema,
});

if run_res == () || !run_res.success || run_res.output == () {
    complete(#{
        ok: false,
        summary: "Architecture gate runner agent failed (no usable output).",
        phase_failed: "Run gate",
    });
}

let run_out = run_res.output;
if run_out.ok != true {
    let err = run_out.error;
    if err == () { err = run_out.stdout_tail; }
    if err == () { err = "validate_architecture.py did not report success"; }
    // Still try to verify if files appeared; fall through only when ok.
    log("Gate command reported failure: " + err.to_string());
}

// =====================================================================
// Phase 2 — independently read the artifacts (fail closed on missing proof)
// =====================================================================
phase("Verify outputs");

let verify_prompt = "";
verify_prompt += "Independently verify the SCD architecture gate outputs in this repo.\n";
verify_prompt += "Use read_file / list_dir only — do not re-run the bench unless files are missing.\n\n";
verify_prompt += "Required reads:\n";
verify_prompt += "1. docs/arch_validation_results.json — parse gates and ok.\n";
verify_prompt += "2. docs/arch_validation_report.md — confirm it exists and has an Overall PASS/FAIL.\n";
verify_prompt += "3. Optionally skim docs/tier0_results.json and docs/phase25_tier1.json if the JSON references them.\n\n";
verify_prompt += "For each gate in results.gates (unit_tests, check_findings, tier0_live_bench,\n";
verify_prompt += "tier0_kill_criterion, tier1_flat_vram), record pass true/false.\n";
verify_prompt += "Extract 2–4 headline numbers if present (e.g. 768p/15s speedup, Tier1 N=16 speedup,\n";
verify_prompt += "flat KV cache yes/no). Explain jargon in plain words when you quote it\n";
verify_prompt += "(e.g. \"KV cache (key-value memory for past frames)\").\n\n";
verify_prompt += "Set artifacts_present=true only if BOTH the JSON and Markdown files exist and parse.\n";
verify_prompt += "Set overall_ok to the JSON's top-level ok field (false if file missing or unreadable).\n";
verify_prompt += "evidence must quote concrete paths or field values you actually read — not memory.\n";

let verify_schema = #{
    "type": "object",
    "required": ["artifacts_present", "overall_ok", "gates", "headlines", "evidence", "issues"],
    "properties": #{
        "artifacts_present": #{ "type": "boolean" },
        "overall_ok": #{ "type": "boolean" },
        "gates": #{
            "type": "object",
            "properties": #{
                "unit_tests": #{ "type": "boolean" },
                "check_findings": #{ "type": "boolean" },
                "tier0_live_bench": #{ "type": "boolean" },
                "tier0_kill_criterion": #{ "type": "boolean" },
                "tier1_flat_vram": #{ "type": "boolean" },
            },
        },
        "headlines": #{
            "type": "array",
            "maxItems": 8,
            "items": #{ "type": "string" },
        },
        "evidence": #{ "type": "string" },
        "issues": #{
            "type": "array",
            "maxItems": 12,
            "items": #{ "type": "string" },
        },
    },
};

let verify_res = agent(verify_prompt, #{
    label: "verify-arch-artifacts",
    capability_mode: "read-only",
    output_schema: verify_schema,
});

if verify_res == () || !verify_res.success || verify_res.output == () {
    complete(#{
        ok: false,
        summary: "Could not verify architecture artifacts (verifier agent failed).",
        phase_failed: "Verify outputs",
        run: run_out,
    });
}

let v = verify_res.output;

// Fail closed: missing artifacts or overall_ok false means the gate did not pass.
let artifacts_ok = v.artifacts_present == true;
let overall_ok = artifacts_ok && v.overall_ok == true && run_out.ok == true;

// =====================================================================
// Phase 3 — write a short handoff people can read in /workflows
// =====================================================================
phase("Summarize");

let status_word = if overall_ok { "PASS" } else { "FAIL" };

let report = "";
report += "# SCD architecture validation\n\n";
report += "**Result: " + status_word + "**\n\n";
report += "Command: `" + cmd + "`\n\n";

report += "## Gates\n\n";
if v.gates != () {
    let g = v.gates;
    if g.unit_tests != () { report += "- " + gate_line("unit_tests (code contracts)", g.unit_tests) + "\n"; }
    if g.check_findings != () { report += "- " + gate_line("check_findings (claims vs JSON)", g.check_findings) + "\n"; }
    if g.tier0_live_bench != () { report += "- " + gate_line("tier0_live_bench (optional re-measure)", g.tier0_live_bench) + "\n"; }
    if g.tier0_kill_criterion != () { report += "- " + gate_line("tier0_kill_criterion (SCD faster at 768p/15s)", g.tier0_kill_criterion) + "\n"; }
    if g.tier1_flat_vram != () { report += "- " + gate_line("tier1_flat_vram (KV cache flat across length)", g.tier1_flat_vram) + "\n"; }
} else {
    report += "_Gates object missing from verifier._\n";
}

report += "\n## Headlines\n\n";
if v.headlines != () {
    for h in v.headlines {
        report += "- " + h + "\n";
    }
} else {
    report += "_None extracted._\n";
}

report += "\n## Issues\n\n";
if v.issues != () && v.issues.len() > 0 {
    for issue in v.issues {
        report += "- " + issue + "\n";
    }
} else if overall_ok {
    report += "- None.\n";
} else {
    report += "- Gate failed; see docs/arch_validation_report.md for detail.\n";
}

report += "\n## Artifacts\n\n";
report += "- `docs/arch_validation_results.json` — machine-readable gates\n";
report += "- `docs/arch_validation_report.md` — human report\n";
if v.evidence != () && v.evidence != "" {
    report += "\n## Verifier evidence\n\n";
    report += v.evidence + "\n";
}

let scratch_path = write_scratch_file("arch_validation_handoff.md", report);

log("Architecture validation " + status_word + " — see " + scratch_path);

complete(#{
    ok: overall_ok,
    status: status_word,
    command: cmd,
    results_json: "docs/arch_validation_results.json",
    report_md: "docs/arch_validation_report.md",
    handoff: scratch_path,
    headlines: v.headlines,
    issues: v.issues,
    gates: v.gates,
    summary: "SCD architecture validation " + status_word
        + ". Results: docs/arch_validation_results.json and docs/arch_validation_report.md.",
});
