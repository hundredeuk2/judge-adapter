"""FastChat reference mapping, the parts that need no FastChat install (synthetic files only, model-free).

The full host flow with a real FastChat checkout is in test_fastchat_reference.py (skipped without one)."""
from __future__ import annotations

import hashlib
import json

import pytest

from judge_adapter import fastchat_runner as RF
from judge_adapter import user_prompt as U
from judge_adapter.hosts import fastchat as B

BENCH = 'synthetic_bench'
QUESTIONS = [{'question_id': 1, 'category': 'writing', 'turns': ['Write a haiku.', 'Make it rhyme.']},
             {'question_id': 2, 'category': 'math', 'turns': ['What is 2 + 3?', 'And times 4?']},
             {'question_id': 3, 'category': 'coding', 'turns': ['Reverse a string in Python.']}]
HOST = ['--mode', 'pairwise-baseline', '--bench-name', BENCH, '--judge-model', 'judge-x', '--baseline-model', 'm-y',
        '--model-list', 'm-x']


def row(qid, turns):
    return {'question_id': qid, 'answer_id': f'ref-{qid}', 'model_id': 'my-reference',
            'choices': [{'index': 0, 'turns': list(turns)}], 'tstamp': 0.0}


def jsonl(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows).encode('utf-8')


def workdir(tmp_path):
    b = tmp_path / 'w' / 'data' / BENCH
    (b / 'reference_answer').mkdir(parents=True)
    (b / 'model_answer').mkdir()
    (b / 'question.jsonl').write_bytes(jsonl(QUESTIONS))
    (b / 'reference_answer' / 'judge-x.jsonl').write_bytes(jsonl([row(2, ['DECOY', 'DECOY']), row(3, ['DECOY'])]))
    (b / 'model_answer' / 'm-x.jsonl').write_text('{}\n')
    (tmp_path / 'w' / 'data' / 'judge_prompts.jsonl').write_text('{}\n')
    return tmp_path / 'w'


def test_good_mapping_covers_only_reference_categories():
    m = B.build_reference_map(QUESTIONS, B.load_reference_rows(jsonl([row(2, ['5', '20']), row(3, ['s[::-1]']),
                                                                       row(99, ['ignored'])])))
    assert m.turns == {2: ('5', '20'), 3: ('s[::-1]',)} and m.record['mapped_question_ids'] == [2, 3]
    rec = m.host_record(2)
    assert rec['turns_sha256'] == hashlib.sha256(json.dumps(['5', '20']).encode()).hexdigest()


@pytest.mark.parametrize('rows,needle', [
    ([row(3, ['x'])], "question 2 (math): no reference row"),
    ([row(2, ['5']), row(3, ['x'])], 'reference has 1 turns, question has 2'),
    ([row(2, ['5', '  ']), row(3, ['x'])], 'empty or not text'),
    ([row(2, ['5', 'y' * (U.LIMITS['reference_entry_bytes'] + 1)]), row(3, ['x'])], 'UTF-8 bytes (not truncated)'),
])
def test_bad_mapping_is_refused_and_nothing_is_filled(rows, needle):
    with pytest.raises(B.ReferenceMappingError, match=needle.replace('(', r'\(').replace(')', r'\)')):
        B.build_reference_map(QUESTIONS, B.load_reference_rows(jsonl(rows)))


def test_reference_rows_parse_errors():
    with pytest.raises(B.ReferenceMappingError, match='duplicate question_id'):
        B.load_reference_rows(jsonl([row(2, ['a', 'b']), row(2, ['a', 'b'])]))
    with pytest.raises(B.ReferenceMappingError, match='not a FastChat answer row'):
        B.load_reference_rows(b'{"question_id": 2, "turns": ["a"]}\n')


def test_prepare_reference_uses_only_the_named_file(tmp_path):
    w = workdir(tmp_path)
    refs = tmp_path / 'mine.jsonl'
    refs.write_bytes(jsonl([row(2, ['5', '20']), row(3, ['s[::-1]'])]))
    refmap, data = RF.prepare_reference(w, HOST, reference_file=str(refs))
    assert data == refs.read_bytes() and refmap.turns[2] == ('5', '20')     # the judge-named decoy is not used
    r = refmap.record
    assert r['inferred_from_judge_name'] is False and r['auto_fill'] is False and r['host_lookup_key'] == 'judge-x'
    assert r['ignored_workdir_reference_files'] == ['judge-x.jsonl']
    assert r['supported_host_revision'] == B.SUPPORTED_HOST_REVISION
    assert r['host_template_replacement']['pair-math-v1'] == U.PROMPT_VERSION_USER
    assert RF.prepare_reference(w, HOST) is None                              # no mapping -> nothing inferred


@pytest.mark.parametrize('kw,needle', [
    ({'reference_file': '/nonexistent/refs.jsonl'}, 'reference file not found'),
    ({'reference_model': 'missing-model'}, 'reference file not found'),
    ({'reference_model': '../escape'}, 'plain file stem'),
    ({'reference_file': 'x', 'reference_model': 'y'}, 'not both'),
])
def test_prepare_reference_option_errors(tmp_path, kw, needle):
    with pytest.raises(B.ReferenceMappingError, match=needle):
        RF.prepare_reference(workdir(tmp_path), HOST, **kw)


def test_stage_holds_only_the_mapped_file_and_cleanup_keeps_the_workdir(tmp_path):
    w = workdir(tmp_path)
    stage = tmp_path / 'stage'
    stage.mkdir()
    before = {str(p.relative_to(w)): p.read_bytes() for p in w.rglob('*') if p.is_file()}
    RF.stage_workdir(w, stage, BENCH, 'judge-x', b'MAPPED\n')
    sref = stage / 'data' / BENCH / 'reference_answer'
    assert [p.name for p in sref.iterdir()] == ['judge-x.jsonl'] and (sref / 'judge-x.jsonl').read_bytes() == b'MAPPED\n'
    assert (stage / 'data' / BENCH / 'question.jsonl').is_symlink()
    RF._remove_stage(stage)
    after = {str(p.relative_to(w)): p.read_bytes() for p in w.rglob('*') if p.is_file()}
    assert not stage.exists() and after == before and (w / 'data' / BENCH / 'model_judgment').is_dir()
