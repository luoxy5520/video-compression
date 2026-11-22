"""Build and run an independent diagnostic JM with optional phase probes."""
import difflib
import json
import re
import shutil

from smoke import ROOT, ENV, record, run, sha, write_json

HEADER = """#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static int lmp_phase_enabled;
static double lmp_mb_s, lmp_entropy_s;
double lmp_transform_s, lmp_residual_s;
unsigned long long lmp_p16_tests, lmp_skip_checks;
double lmp_phase_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
int lmp_phase_on(void) { return lmp_phase_enabled; }
static void lmp_phase_report(void)
{
  fprintf(stderr, "LMP_PHASE {\\"macroblock_s\\":%.9f,\\"entropy_s\\":%.9f,"
          "\\"transform_s\\":%.9f,\\"residual_s\\":%.9f,\\"p16_tests\\":%llu,"
          "\\"skip_checks\\":%llu}\\n", lmp_mb_s, lmp_entropy_s,
          lmp_transform_s, lmp_residual_s, lmp_p16_tests, lmp_skip_checks);
}
static void lmp_phase_init(void)
{
  static int initialized;
  if (!initialized)
  {
    lmp_phase_enabled = getenv("LMP_DETAIL") != NULL;
    if (lmp_phase_enabled)
      atexit(lmp_phase_report);
    initialized = 1;
  }
}
"""
EXTERN = """extern int lmp_phase_on(void);
extern double lmp_phase_now(void), lmp_transform_s, lmp_residual_s;
extern unsigned long long lmp_p16_tests, lmp_skip_checks;
"""
TRANSFORM = re.compile(r"^int (residual_transform_quant_\w+)\(([^\n]+)\)\n\{", re.M)


def transform_probes(code):
    wrappers, declarations = [], []
    for match in TRANSFORM.finditer(code):
        name, params = match.groups()
        args = ", ".join(re.search(r"\w+$", arg.strip())[0] for arg in params.split(","))
        declarations.append(f"int {name}({params});")
        wrappers.append(f"""
int {name}({params})
{{
  double t;
  int result;
  if (!lmp_phase_on())
    return lmp_unprofiled_{name}({args});
  t = lmp_phase_now();
  result = lmp_unprofiled_{name}({args});
  lmp_transform_s += lmp_phase_now() - t;
  return result;
}}
""")
    assert wrappers
    anchor = '#include "global.h"'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + "\n" + "\n".join(declarations))
    return EXTERN + TRANSFORM.sub(r"static int lmp_unprofiled_\1(\2)\n{", code) + "".join(wrappers)


def add_probes(src):
    updates = {}
    path = src / "slice.c"
    code = path.read_text()
    original = """        currSlice->encode_one_macroblock (currMB);
        end_encode_one_macroblock(currMB);
        write_macroblock (currMB, 1);"""
    assert code.count(original) == 1
    updates[path] = HEADER + code.replace(original, """        double lmp_t = 0;
        lmp_phase_init();
        if (lmp_phase_enabled) lmp_t = lmp_phase_now();
        currSlice->encode_one_macroblock (currMB);
        end_encode_one_macroblock(currMB);
        if (lmp_phase_enabled)
        {
          lmp_mb_s += lmp_phase_now() - lmp_t;
          lmp_t = lmp_phase_now();
        }
        write_macroblock (currMB, 1);
        if (lmp_phase_enabled) lmp_entropy_s += lmp_phase_now() - lmp_t;""")
    for name in ("block.c", "transform8x8.c"):
        path = src / name
        updates[path] = transform_probes(path.read_text())
    path = src / "macroblock.c"
    code, count = re.subn(r"(?m)^( +)(compute_residue\s*\([^\n]+;)$",
                          r"""\1{ double t = lmp_phase_on() ? lmp_phase_now() : 0;
\1  \2
\1  if (lmp_phase_on()) lmp_residual_s += lmp_phase_now() - t;
\1}""", path.read_text())
    assert count >= 8
    updates[path] = EXTERN + code
    path = src / "md_low.c"
    code = path.read_text()
    anchor = "        for (cost=0, block=0; block<(mode==1?1:2); block++)"
    assert code.count(anchor) == 1
    code = code.replace(anchor, "        if (lmp_phase_on() && mode == 1) ++lmp_p16_tests;\n" + anchor)
    anchor = "    //===== check for SKIP mode ====="
    assert code.count(anchor) == 1
    updates[path] = EXTERN + code.replace(anchor, anchor + "\n    if (lmp_phase_on()) ++lmp_skip_checks;")
    patch = []
    for path, new in updates.items():
        before = path.read_text()
        label = f"lencod/src/{path.name}"
        patch.extend(difflib.unified_diff(before.splitlines(True), new.splitlines(True),
                                          f"a/{label}", f"b/{label}"))
        path.write_text(new)
    (ROOT / "configs/jm_phase_timing.patch").write_text("".join(patch))


def main():
    detail = ROOT / "jm-detail"
    shutil.copytree(ROOT / "jm", detail, ignore=shutil.ignore_patterns("obj", "*.exe"))
    add_probes(detail / "lencod/src")
    record(run(["make", "-j4", "CFLAGS=-fcommon"], detail / "lencod", "build-phase-detail"))
    encoder = detail / "bin/lencod.exe"
    base = [json.loads(line) for line in (ROOT / "results/repeats.jsonl").read_text().splitlines()]
    output = []
    for row in base:
        if row["arm"] != "C0":
            continue
        group = ROOT / "logs/runs" / row["sample"] / "detail"
        group.mkdir(exist_ok=True)
        bit, recon = group / "stream.264", group / "recon.yuv"
        params = dict(row["params"], OutputFile=str(bit), ReconFile=str(recon))
        cfgs = list(row["config_sha256"])
        encoder_args = [encoder, "-d", cfgs[0], "-f", cfgs[1]]
        for key, value in params.items():
            encoder_args.extend(("-p", f"{key} = {value}"))
        for active in (False, True) if row["warmup"] else (True,):
            stem = f'{row["repeat"]}-{"enabled" if active else "disabled"}'
            usage = group / f"{stem}.rss_kib"
            env = dict(ENV, LMP_MODE="0")
            if active:
                env["LMP_DETAIL"] = "1"
            wall = record(run(["/usr/bin/time", "-f", "%M", "-o", usage, *encoder_args],
                              group, stem, env, [bit, recon, usage]))
            assert sha(bit) == row["bitstream_sha256"]
            stderr = (group / f"{stem}.stderr").read_text()
            details = dict(sample=row["sample"], repeat=row["repeat"], probes_enabled=active,
                           wall_s=wall, peak_rss_kib=int(usage.read_text()),
                           encoder_sha256=sha(encoder), bitstream_sha256=sha(bit))
            for label in ("LMP_PHASE", "LMP_PROFILE"):
                match = re.search(rf"^{label} (\{{.*\}})$", stderr, re.M)
                if label == "LMP_PHASE" and not active:
                    assert match is None
                    continue
                assert match, (group, label)
                details.update(json.loads(match[1]))
            if active:
                details["mode_and_other_mb_s"] = (
                    details["macroblock_s"] - details["integer_s"] - details["subpixel_s"]
                    - details["transform_s"] - details["residual_s"]
                )
            output.append(details)
            print(json.dumps(details), flush=True)
    write_json(ROOT / "results/phase_detail.json", output)


if __name__ == "__main__":
    main()
