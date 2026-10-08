# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
#
# OPTIONAL module (extra "fastchat"). Runs FastChat llm_judge (https://github.com/lm-sys/FastChat, Apache License
# 2.0, commit 587d5cfa1609a43d192cedb8441cac3c17db105d) through the adapter. It contains no FastChat source code;
# parse_host_args mirrors the option names and defaults of fastchat/llm_judge/gen_judgment.py at that commit.
# demo_workdir copies FastChat's own data files from the user's FastChat checkout or installation at run time;
# no FastChat or MT-Bench data is bundled with this package. See NOTICE.
"""Run FastChat llm_judge through the adapter. The host's own gen_judgment.py __main__ is executed unchanged.

  judge-adapter-fastchat demo-workdir --workdir W --fastchat-root FASTCHAT_CHECKOUT
  judge-adapter-fastchat judge --config config.fake.json --workdir W --confirm -- --mode pairwise-baseline \
      --bench-name adapter_demo --judge-model judge-adapter --baseline-model model-y --model-list model-x
  judge-adapter-fastchat show --workdir W -- --mode pairwise-baseline --bench-name adapter_demo \
      --judge-model judge-adapter --baseline-model model-y

Reference categories (math, reasoning, coding, arena-hard-200) need an EXPLICIT reference mapping:
  ... judge ... --reference-file my_refs.jsonl -- --mode pairwise-baseline ...     # a file the user provides
  ... judge ... --reference-model gpt-4 -- ...      # = W/data/<bench>/reference_answer/gpt-4.jsonl
The host looks the reference up by --judge-model name. The wrapper never reuses a file found under that name: it
runs the host in a temporary staged workdir (built with symlinks) whose reference_answer/ holds ONLY the mapped
file, copied under <judge-model>.jsonl, and records the mapping in <trace_dir>/fastchat_reference_mapping.jsonl.
Missing rows or a turn-count mismatch are refused before the host starts. Without a mapping, reference-category
requests stay unsupported_input (0 calls).

Everything after "--" goes to the host script as its own arguments. transport "real" also needs --allow-real.
Host --mode must be given explicitly and be pairwise-baseline or pairwise-all (the only modes whose judge calls go
through the patched run_judge_pair). Host "single" (its default) and anything else are refused BEFORE the host runs:
host single grading calls its own judge client, which would bypass the fake transport and --allow-real.

--fastchat-root: a FastChat source checkout to put on sys.path (and to read data/ from). Without it the installed
``fastchat`` package is used; demo-workdir then needs fastchat/llm_judge/data inside that installation.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import hashlib
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_FASTCHAT = None              # None -> the installed fastchat package (no checkout path is built in)
DEMO_BENCH = 'adapter_demo'
SUPPORTED_HOST_MODES = ('pairwise-baseline', 'pairwise-all')
DEMO_QUESTIONS = (81, 82, 111)       # two writing questions + one math question (reference-based judge)
DEMO_ANSWERS = {                     # synthetic answers; markers drive the fake judge (see fakes.py)
    'model-x': {81: ['[[GOOD]] A Hawaii travel post by X.', 'FAKE:POSITION All sentences start with A, by X.'],
                82: ['An email draft by X.', 'FAKE:FORMAT A self-critique by X.'],
                111: ['The area is 3, by X.', 'The circle area, by X.']},
    'model-y': {81: ['A Hawaii travel post by Y.', 'All sentences start with A, by Y.'],
                82: ['An email draft by Y.', 'A self-critique by Y.'],
                111: ['The area is 3, by Y.', 'The circle area, by Y.']},
}


def _host_path(fastchat_root):
    if fastchat_root is None:               # use the installed fastchat package
        return
    sys.dont_write_bytecode = True          # never write __pycache__ into the user's checkout
    p = str(Path(fastchat_root).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def host_data_dir(fastchat_root=None):
    """fastchat/llm_judge/data of the checkout, or of the installed fastchat package."""
    if fastchat_root is not None:
        d = Path(fastchat_root) / 'fastchat' / 'llm_judge' / 'data'
    else:
        try:
            import fastchat
        except ImportError:
            raise RuntimeError('fastchat is not installed; pip install "judge-adapter[fastchat]" or pass '
                               '--fastchat-root') from None
        d = Path(fastchat.__file__).resolve().parent / 'llm_judge' / 'data'
    if not (d / 'judge_prompts.jsonl').is_file():
        raise RuntimeError(f'FastChat llm_judge data not found at {str(d)!r}; pass --fastchat-root pointing to a '
                           'FastChat source checkout')
    return d


def demo_workdir(workdir, fastchat_root=DEFAULT_FASTCHAT, answers=None, bench=DEMO_BENCH, judge_label='judge-adapter'):
    """Build data/ for the host from its own files (questions, judge prompts, reference) + synthetic answers.

    reference_answer/ gets the host's MT-bench reference twice: under its real name gpt-4.jsonl (for an explicit
    ``--reference-model gpt-4``) and under ``<judge_label>.jsonl``, only so the host's own check_data passes in the
    default (no mapping) run. That judge-named copy is never used as a reference by the adapter."""
    src = host_data_dir(fastchat_root)
    w = Path(workdir) / 'data'
    (w / bench / 'model_answer').mkdir(parents=True, exist_ok=True)
    (w / bench / 'reference_answer').mkdir(parents=True, exist_ok=True)
    (w / 'judge_prompts.jsonl').write_text((src / 'judge_prompts.jsonl').read_text(encoding='utf-8'), encoding='utf-8')
    answers = answers or DEMO_ANSWERS
    qids = sorted({q for a in answers.values() for q in a})
    qs = [json.loads(l) for l in (src / 'mt_bench' / 'question.jsonl').read_text(encoding='utf-8').splitlines() if l]
    (w / bench / 'question.jsonl').write_text(
        ''.join(json.dumps(q) + '\n' for q in qs if q['question_id'] in qids), encoding='utf-8')
    refs = [json.loads(l) for l in (src / 'mt_bench' / 'reference_answer' / 'gpt-4.jsonl').read_text(
        encoding='utf-8').splitlines() if l]
    ref_text = ''.join(json.dumps(r) + '\n' for r in refs if r['question_id'] in qids)
    (w / bench / 'reference_answer' / f'{judge_label}.jsonl').write_text(ref_text, encoding='utf-8')
    (w / bench / 'reference_answer' / 'gpt-4.jsonl').write_text(ref_text, encoding='utf-8')
    for model, per_q in answers.items():
        rows = [{'question_id': q, 'answer_id': f'ans-{model}-q{q}', 'model_id': model,
                 'choices': [{'index': 0, 'turns': list(t)}], 'tstamp': 0.0} for q, t in sorted(per_q.items())]
        (w / bench / 'model_answer' / f'{model}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return w


class HostArgsRefused(ValueError):
    pass


HOST_DEFAULTS = {'bench_name': 'mt_bench', 'judge_file': 'data/judge_prompts.jsonl', 'judge_model': 'gpt-4'}


def parse_host_args(host_args):
    """Parse the host arguments with the host's own option set (gen_judgment.py at the pinned commit, same argparse
    prefix rules and defaults) and allow only the pairwise modes. Runs before the host is imported or started."""
    ap = argparse.ArgumentParser(prog='gen_judgment.py', add_help=False)
    for opt in ('--bench-name', '--judge-file', '--judge-model', '--baseline-model', '--mode'):
        ap.add_argument(opt, type=str, default=HOST_DEFAULTS.get(opt[2:].replace('-', '_')))
    ap.add_argument('--model-list', type=str, nargs='+')
    ap.add_argument('--parallel', type=int)
    ap.add_argument('--first-n', type=int)
    ap.add_argument('-h', '--help', action='store_true')
    try:
        a, unknown = ap.parse_known_args(list(host_args))
    except SystemExit:
        raise HostArgsRefused(f'host arguments could not be parsed: {list(host_args)}') from None
    if unknown or a.help:
        raise HostArgsRefused(f'unsupported host arguments {unknown or ["--help"]}; nothing was run')
    if a.mode not in SUPPORTED_HOST_MODES:
        shown = 'missing (the host default is "single")' if a.mode is None else repr(a.mode)
        raise HostArgsRefused(f'host --mode {shown} is not supported; use one of {SUPPORTED_HOST_MODES}. Host single '
                              'grading would call its own judge client outside the adapter (no fake transport, no '
                              '--allow-real). Nothing was run')
    return a


def check_host_args(host_args):
    return parse_host_args(host_args).mode


# --- explicit reference mapping (host categories that need a reference) ---------------------------------------------
def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def prepare_reference(workdir, host_args, reference_file=None, reference_model=None):
    """Validate an explicit mapping BEFORE the adapter, the host import or the host run. -> (ReferenceMap, bytes) or
    None when no mapping is given. Never infers a reference from --judge-model and never fills a gap."""
    from .hosts import fastchat as binding
    if reference_file is None and reference_model is None:
        return None
    if reference_file is not None and reference_model is not None:
        raise binding.ReferenceMappingError('give --reference-file or --reference-model, not both')
    a = parse_host_args(host_args)
    w = Path(workdir).resolve()
    for name, what in ((a.judge_model, '--judge-model'), (a.bench_name, '--bench-name'),
                       (reference_model, '--reference-model')):
        if name is not None and (not name or '/' in name or '\\' in name or name.startswith('.')):
            raise binding.ReferenceMappingError(f'{what} {name!r} must be a plain file stem (no path) when a '
                                                'reference is mapped')
    if a.judge_file and not Path(a.judge_file).is_absolute():
        jf = (w / a.judge_file).resolve()
        if w not in jf.parents:
            raise binding.ReferenceMappingError(f'relative --judge-file {a.judge_file!r} points outside the workdir; '
                                                'pass an absolute path (the host runs in a staged copy)')
    bench_dir = w / 'data' / a.bench_name
    qfile = bench_dir / 'question.jsonl'
    if not qfile.is_file():
        raise binding.ReferenceMappingError(f'host question file not found: {qfile}')
    if reference_model is not None:
        src, kind = bench_dir / 'reference_answer' / f'{reference_model}.jsonl', 'reference_model'
    else:
        src, kind = Path(reference_file).expanduser(), 'reference_file'
        if not src.is_absolute():
            src = Path.cwd() / src
    src = src.resolve()
    if not src.is_file():
        raise binding.ReferenceMappingError(f'reference file not found: {src}')
    data = src.read_bytes()
    questions = _read_jsonl(qfile)
    if a.first_n:
        questions = questions[:a.first_n]             # same subset the host judges
    ref_dir = bench_dir / 'reference_answer'
    record = {
        'schema': 'judge-adapter-fastchat-reference-mapping/v1', 'host': 'fastchat.llm_judge',
        'supported_host_revision': binding.SUPPORTED_HOST_REVISION, 'bench_name': a.bench_name, 'host_mode': a.mode,
        'source_kind': kind, 'reference_model': reference_model, 'source_path': str(src),
        'source_sha256': hashlib.sha256(data).hexdigest(), 'source_bytes': len(data),
        'host_lookup_key': a.judge_model,
        'staged_as': f'data/{a.bench_name}/reference_answer/{a.judge_model}.jsonl',
        'staged_note': 'temporary staged workdir; reference_answer/ holds only this byte copy; removed after the run',
        'ignored_workdir_reference_files': sorted(p.name for p in ref_dir.glob('*.jsonl')) if ref_dir.is_dir() else [],
        'reference_categories': list(binding.REFERENCE_CATEGORIES),
        'inferred_from_judge_name': False, 'auto_fill': False,
        'host_template_replacement': dict(binding.HOST_TEMPLATE_REPLACEMENT),
    }
    return binding.build_reference_map(questions, binding.load_reference_rows(data), record), data


def stage_workdir(workdir, stage, bench, judge_model, reference_bytes):
    """Mirror the user's workdir with symlinks, except data/<bench>/reference_answer/, which holds ONLY the mapped
    file as <judge_model>.jsonl (byte copy). model_judgment/ links to the user's folder, so host output lands there."""
    w, s = Path(workdir).resolve(), Path(stage)

    def mirror(src, dst, skip):
        for e in sorted(src.iterdir()):
            if e.name not in skip:
                (dst / e.name).symlink_to(e, target_is_directory=e.is_dir())
    mirror(w, s, {'data'})
    (s / 'data').mkdir()
    mirror(w / 'data', s / 'data', {bench})
    sb = s / 'data' / bench
    sb.mkdir()
    mirror(w / 'data' / bench, sb, {'reference_answer', 'model_judgment'})
    out = w / 'data' / bench / 'model_judgment'
    out.mkdir(exist_ok=True)
    (sb / 'model_judgment').symlink_to(out, target_is_directory=True)
    (sb / 'reference_answer').mkdir()
    (sb / 'reference_answer' / f'{judge_model}.jsonl').write_bytes(reference_bytes)
    return s


def _remove_stage(stage):
    """Unlink every symlink first (never follow them into the user's workdir), then remove the staged folder."""
    for root, dirs, files in os.walk(stage, topdown=True, followlinks=False):
        for n in files + list(dirs):
            p = os.path.join(root, n)
            if os.path.islink(p):
                os.unlink(p)
                if n in dirs:
                    dirs.remove(n)
    shutil.rmtree(stage, ignore_errors=False)


def build_adapter(cfg, workdir, transport=None, key_provider=None):
    """fake -> injected fake transport + dummy key (no key source read); real -> network transport + lazy key from
    the explicit env file (cfg.env_file) or else the environment."""
    from .core import JudgeAdapter, build_backend
    trace_dir = Path(cfg.trace_dir)
    if not trace_dir.is_absolute():
        trace_dir = Path(workdir).resolve() / trace_dir
    cfg = dataclasses.replace(cfg, trace_dir=str(trace_dir.resolve()),
                              env_file=str(Path(cfg.env_file).resolve()) if cfg.env_file else None)
    fake = None
    if cfg.transport == 'fake' and transport is None:
        from .fakes import DUMMY_KEY, fake_for
        fake = fake_for(cfg.backend)
        transport, key_provider = fake.transport(), (lambda: DUMMY_KEY)
    return JudgeAdapter(cfg, build_backend(cfg, transport=transport, key_provider=key_provider)), fake


def run_host_module(module, workdir, host_args, confirm=False):
    if module.endswith('gen_judgment'):
        check_host_args(host_args)           # before chdir / import / run: unsupported modes never start the host
    old_cwd, old_argv, old_stdin = os.getcwd(), sys.argv, sys.stdin
    os.chdir(workdir)
    sys.argv = [module.rsplit('.', 1)[-1] + '.py', *host_args]
    if confirm:
        sys.stdin = io.StringIO('\n')        # answers the host's "Press Enter to confirm..." prompt
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    finally:
        os.chdir(old_cwd)
        sys.argv, sys.stdin = old_argv, old_stdin


def judge(cfg, workdir, host_args, fastchat_root=DEFAULT_FASTCHAT, confirm=False, transport=None, key_provider=None,
          reference_file=None, reference_model=None):
    """reference_file / reference_model: explicit reference mapping (see module doc). Both None -> run as before."""
    a = parse_host_args(host_args)           # before the adapter, the host path or the host import
    prepared = prepare_reference(workdir, host_args, reference_file, reference_model)   # refusals: nothing started
    _host_path(fastchat_root)
    from .hosts import fastchat as binding
    adapter, fake = build_adapter(cfg, workdir, transport, key_provider)
    stage, restore = None, None
    try:
        if prepared is None:
            restore = binding.install(adapter)
            run_host_module('fastchat.llm_judge.gen_judgment', workdir, host_args, confirm)
        else:
            refmap, data = prepared
            restore = binding.install(adapter, refmap)
            stage = tempfile.mkdtemp(prefix='judge-adapter-fastchat-ref-')
            stage_workdir(workdir, stage, a.bench_name, a.judge_model, data)
            rec = dict(refmap.record, workdir=str(Path(workdir).resolve()), adapter_trace=str(adapter.trace.trace))
            with open(adapter.trace.dir / 'fastchat_reference_mapping.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            run_host_module('fastchat.llm_judge.gen_judgment', stage, host_args, confirm)
    finally:
        if restore is not None:
            restore()
        adapter.close()
        if stage is not None:
            _remove_stage(stage)
    return adapter, fake


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    host_args = []
    if '--' in argv:
        i = argv.index('--')
        argv, host_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog='judge-adapter-fastchat')
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('demo-workdir')
    d.add_argument('--workdir', required=True)
    d.add_argument('--fastchat-root', default=DEFAULT_FASTCHAT,
                   help='FastChat source checkout (default: the installed fastchat package)')
    j = sub.add_parser('judge')
    j.add_argument('--config', required=True)
    j.add_argument('--workdir', required=True)
    j.add_argument('--fastchat-root', default=DEFAULT_FASTCHAT,
                   help='FastChat source checkout (default: the installed fastchat package)')
    j.add_argument('--confirm', action='store_true', help="answer the host's Enter prompt")
    j.add_argument('--allow-real', action='store_true', help='required when transport is "real" (paid calls)')
    ref = j.add_mutually_exclusive_group()
    ref.add_argument('--reference-file', help='FastChat answer-format JSONL with the reference answers to use for '
                     'math/reasoning/coding questions (explicit; never inferred from --judge-model)')
    ref.add_argument('--reference-model', help='use W/data/<bench>/reference_answer/<NAME>.jsonl as the reference '
                     '(e.g. gpt-4 for the MT-bench reference)')
    s = sub.add_parser('show')
    s.add_argument('--workdir', required=True)
    s.add_argument('--fastchat-root', default=DEFAULT_FASTCHAT,
                   help='FastChat source checkout (default: the installed fastchat package)')
    a = ap.parse_args(argv)
    if a.cmd == 'demo-workdir':
        print(demo_workdir(a.workdir, a.fastchat_root))
        return 0
    if a.cmd == 'show':
        _host_path(a.fastchat_root)
        run_host_module('fastchat.llm_judge.show_result', a.workdir, host_args)
        return 0
    from .contract import AdapterConfig
    cfg = AdapterConfig.load(a.config)
    if cfg.transport == 'real' and not a.allow_real:
        print('transport "real" sends paid requests; pass --allow-real explicitly', file=sys.stderr)
        return 2
    from .hosts.fastchat import ReferenceMappingError
    try:
        check_host_args(host_args)
        prepare_reference(a.workdir, host_args, a.reference_file, a.reference_model)    # refuse before anything
    except (HostArgsRefused, ReferenceMappingError) as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    adapter, fake = judge(cfg, a.workdir, host_args, a.fastchat_root, a.confirm,
                          reference_file=a.reference_file, reference_model=a.reference_model)
    mapping = adapter.trace.dir / 'fastchat_reference_mapping.jsonl'
    print(json.dumps({'adapter_trace': str(adapter.trace.trace), 'raw': str(adapter.trace.raw),
                      'reference_mapping': str(mapping) if (a.reference_file or a.reference_model) else None,
                      'backend': cfg.backend, 'mode': cfg.mode, 'transport': cfg.transport,
                      'fake_http_requests': len(fake.requests) if fake else None}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
